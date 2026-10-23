#!/usr/bin/env python3
"""NP/DE/PJM/FR LIFT Lead Estimator: L=168, H=24, stride=168 hours.

Official local-peak estimator, restricted to the two exogenous -> OT pairs.
H defines complete forecast samples; no future values enter the estimator.
This extracts leads, not the backbone + trained Lead-aware Refiner forecasts.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


L = 168
H = 24
STRIDE = 168
DATASETS = {
    "NP": [" Grid load forecast", " Wind power forecast"],
    "DE": [" Wind power forecast", " Ampirion zonal load forecast"],
    "PJM": [" System load forecast", " Zonal COMED load foecast"],
    "FR": [" Generation forecast", " System load forecast"],
}
ENDOGENOUS = "OT"
UPSTREAM_COMMIT = "79cf8f157bb9a616c9733daf1e9c2cfefdc98c94"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def read_dataset(path, exogenous):
    frame = pd.read_csv(path, dtype={"date": str, "cols": str})
    if list(frame.columns) != ["date", "data", "cols"]:
        raise ValueError("Dataset CSV must have the long-table columns date,data,cols")
    if frame[["date", "cols"]].isna().any().any():
        raise ValueError("Missing date or channel name")
    expected = exogenous + [ENDOGENOUS]
    if set(frame.cols.unique()) != set(expected):
        raise ValueError(f"Expected exactly these dataset channels, including spaces: {expected}")
    times = None
    values = []
    for column in expected:
        group = frame[frame.cols == column]
        dates = pd.to_datetime(group.date, format="mixed", errors="raise").to_numpy(dtype="datetime64[ns]")
        if np.isnat(dates).any():
            raise ValueError(f"Invalid timestamp in {column!r}")
        if times is None:
            times = dates
        elif not np.array_equal(times, dates):
            raise ValueError(f"Time axis does not match for {column!r}")
        values.append(pd.to_numeric(group.data, errors="raise").to_numpy(dtype=np.float64))
    if len(times) < L + H:
        raise ValueError(f"Need at least {L + H} timestamps")
    delta = np.diff(times.astype(np.int64))
    if not (delta == 3_600_000_000_000).all():
        raise ValueError("Dataset must be strictly increasing, duplicate-free and hourly")
    data = np.stack(values, axis=-1)
    if not np.isfinite(data).all():
        raise ValueError("Non-finite dataset values: strict estimator does not impute/drop samples")
    return times, data


def instance_norm(ts, dim=-1):
    # Same operations and epsilon as official models/LIFT.py.
    mu = ts.mean(dim, keepdim=True)
    ts = ts - mu
    std = ((ts ** 2).mean(dim, keepdim=True) + 1e-8) ** 0.5
    return ts / std, mu, std


@torch.no_grad()
def lead_estimator(history):
    """history [B,3,L], channels=[generation,load,OT], float32 CPU.

    Same target-specific entries as upstream cross_corr_coef/estimate_indicator:
    only the broadcast indices are restricted to OT x two exogenous channels.
    Retain masked zero scores, first-maximum ties and the upstream lag=1 fallback.
    """
    if history.ndim != 3 or history.shape[1:] != (3, L):
        raise ValueError(f"Expected [B,3,{L}], got {tuple(history.shape)}")
    normalized, _, _ = instance_norm(history, -1)
    rfft = torch.fft.rfft(normalized, dim=-1)
    cross_corr = torch.fft.irfft(
        rfft[:, 2:3] * torch.conj(rfft[:, :2]), dim=-1, n=L)
    corr_abs = cross_corr.abs()
    mask = ((corr_abs[..., 1:-1] >= corr_abs[..., :-2]) &
            (corr_abs[..., 1:-1] >= corr_abs[..., 2:]))
    masked = cross_corr[..., 1:-1] * mask
    masked = masked / L
    _, shift_index = masked.abs().max(-1)
    score = masked.gather(-1, shift_index.unsqueeze(-1)).squeeze(-1)
    lag = shift_index + 1
    # K=2 retains both requested exogenous channels, ranked as in upstream Top-K.
    rank_order = score.abs().topk(2, dim=-1).indices
    rank = torch.empty_like(rank_order)
    rank.scatter_(1, rank_order, torch.arange(1, 3, device=history.device).expand(len(history), -1))
    return lag, score, rank


def pearson(a, b):
    """Separate diagnostic: actual segments, no FFT wrap and no imputation."""
    ac = a - a.mean(axis=-1, keepdims=True)
    bc = b - b.mean(axis=-1, keepdims=True)
    denom = np.sqrt((ac * ac).sum(-1) * (bc * bc).sum(-1))
    return np.divide((ac * bc).sum(-1), denom,
                     out=np.full_like(denom, np.nan), where=denom > 0)


def validate_non_circular(data, starts, lag):
    """Audit the selected d only. Never alter/reselect LIFT lag or score."""
    count = len(starts)
    target = data[starts[:, None] + np.arange(L), 2]
    history_score = np.full((count, 2), np.nan)
    overlap_score = np.full((count, 2), np.nan)
    history_n = np.zeros((count, 2), dtype=int)
    for j in range(2):
        good = starts >= lag[:, j]
        idx = starts[good, None] - lag[good, j, None] + np.arange(L)
        history_score[good, j] = pearson(data[idx, j], target[good])
        history_n[good, j] = L
        # Group by chosen lag, preserving the actual L-d overlap even if only 2.
        for d in np.unique(lag[:, j]):
            selected = lag[:, j] == d
            ss = starts[selected]
            xx = data[ss[:, None] + np.arange(L - d), j]
            yy = data[ss[:, None] + d + np.arange(L - d), 2]
            overlap_score[selected, j] = pearson(xx, yy)
    return history_score, history_n, overlap_score


def run_dataset(dataset, source_path, output, batch_size, overwrite):
    EXOGENOUS = DATASETS[dataset]
    args = SimpleNamespace(input=Path(source_path), output=Path(output),
                           batch_size=batch_size, overwrite=overwrite)
    source = args.input.expanduser().resolve()
    dest = args.output.expanduser().resolve()
    if dest == source.parent or source.parent in dest.parents:
        raise ValueError("Output must be outside the source data directory")
    if dest.exists() and not args.overwrite:
        raise ValueError("Output exists: choose another directory or use --overwrite")
    torch.set_num_threads(1)
    started = time.monotonic()
    times, data = read_dataset(source, EXOGENOUS)
    all_starts = np.arange(0, len(times) - L - H + 1, STRIDE)
    n = len(all_starts)
    dest.mkdir(parents=True, exist_ok=True)
    meta = dict(status="running", dataset=dataset, history_length=L, forecast_horizon=H,
        stride=STRIDE, lag_candidates=[1, L - 2], endogenous=ENDOGENOUS,
        exogenous=EXOGENOUS, retained_exogenous_K=2,
        direction="exogenous[t-d] corresponds to OT[t]; positive d means exogenous leads",
        estimator="official LIFT circular FFT, local peak mask, same instance_norm and reduction",
        scope="Lead Estimator only; no backbone/refiner training or 24-step value prediction",
        scaling="official instance_norm on raw dataset history; no global StandardScaler or train/test split",
        numeric_dtype="torch.float32", normalization_epsilon=1e-8,
        source=dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    time_points=len(times), interval_seconds=3600,
                    start=str(times[0]), end=str(times[-1])),
        upstream_commit=UPSTREAM_COMMIT,
        python=sys.version, torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__)
    write_json(dest / "metadata.json", meta)
    result_parts, check_parts = [], []
    source32 = torch.as_tensor(data, dtype=torch.float32)
    try:
        for begin in range(0, n, args.batch_size):
            starts = all_starts[begin:begin + args.batch_size]
            idx = torch.as_tensor(starts[:, None] + np.arange(L), dtype=torch.long)
            history = source32[idx].permute(0, 2, 1)
            lag_t, score_t, rank_t = lead_estimator(history)
            lag, score, rank = lag_t.numpy(), score_t.numpy(), rank_t.numpy()
            history_end = starts + L - 1
            forecast_start = starts + L
            forecast_end = starts + L + H - 1
            base = dict(window_id=np.repeat(starts, 2),
                history_start_index=np.repeat(starts, 2),
                history_end_index=np.repeat(history_end, 2),
                forecast_start_index=np.repeat(forecast_start, 2),
                forecast_end_index=np.repeat(forecast_end, 2),
                history_start=np.repeat(times[starts], 2),
                history_end=np.repeat(times[history_end], 2),
                forecast_start=np.repeat(times[forecast_start], 2),
                forecast_end=np.repeat(times[forecast_end], 2),
                exogenous=np.tile(EXOGENOUS, len(starts)), endogenous=ENDOGENOUS)
            result_parts.append(pd.DataFrame({**base,
                "lag_steps":lag.ravel(), "lag_hours":lag.ravel(),
                "score":score.ravel(), "abs_score":np.abs(score).ravel(),
                "exogenous_rank":rank.ravel(),
                "upstream_zero_score_fallback":(score == 0).ravel()}))
            hist, hist_n, overlap = validate_non_circular(data, starts, lag)
            check_parts.append(pd.DataFrame(dict(window_id=np.repeat(starts, 2),
                exogenous=np.tile(EXOGENOUS, len(starts)), endogenous=ENDOGENOUS,
                lift_lag_steps=lag.ravel(), history_pearson=hist.ravel(),
                history_n=hist_n.ravel(), overlap_pearson=overlap.ravel(),
                overlap_n=(L - lag).ravel())))
        result = pd.concat(result_parts, ignore_index=True)
        checks = pd.concat(check_parts, ignore_index=True)
        for frame, filename in [(result, f"{dataset}_lift_window_lags.csv.gz"),
                                (checks, f"{dataset}_non_circular_checks.csv.gz")]:
            part = dest / (filename + ".part")
            with gzip.open(part, "wt", encoding="utf-8", newline="") as out:
                frame.to_csv(out, index=False, float_format="%.10g", date_format="%Y-%m-%d %H:%M:%S")
            part.replace(dest / filename)
        summaries = []
        for j, name in enumerate(EXOGENOUS):
            pair = result[result.exogenous == name]
            counts = Counter(pair.lag_steps.astype(int).tolist())
            summaries.append(dict(exogenous=name, endogenous=ENDOGENOUS,
                windows=len(pair), mean_lag_hours=float(pair.lag_hours.mean()),
                median_lag_hours=float(pair.lag_hours.median()), mean_abs_score=float(pair.abs_score.mean()),
                median_abs_score=float(pair.abs_score.median()),
                most_common_lags=[dict(lag_steps=d, windows=c, fraction=c / n)
                                  for d, c in counts.most_common(10)],
                zero_score_fallback_windows=int(pair.upstream_zero_score_fallback.sum()),
                lag_histogram=dict(sorted(counts.items()))))
        write_json(dest / "summary.json", summaries)
        from plot_lags import plot_distributions
        plot_paths = plot_distributions(result, dest / "plots", dataset=dataset, exogenous=EXOGENOUS, stride_hours=STRIDE)
        meta.update(status="complete", windows=n, rows=len(result),
                    plot_units="hours", plot_files=plot_paths,
                    first_forecast_start=str(times[L]), last_forecast_end=str(times[all_starts[-1] + L + H - 1]),
                    elapsed_seconds=round(time.monotonic() - started, 3))
        write_json(dest / "metadata.json", meta)
        print(json.dumps({k:meta[k] for k in ["dataset", "history_length", "forecast_horizon",
                         "stride", "lag_candidates", "windows", "rows", "elapsed_seconds"]},
                         ensure_ascii=False, indent=2))
    except Exception as exc:
        meta.update(status="failed", error=str(exc))
        write_json(dest / "metadata.json", meta)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", type=str.upper, choices=list(DATASETS),
                        default=list(DATASETS), help="Default: NP DE PJM FR")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "Downloads/dataset",
                        help="Folder containing CSVs, or its parent containing forecasting/")
    parser.add_argument("--input", type=Path, help="Exact CSV path; requires exactly one dataset")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Duplicate datasets")
    if args.input and len(args.datasets) != 1:
        parser.error("--input requires exactly one --datasets name")
    root = args.data_dir.expanduser().resolve()
    jobs = []
    for dataset in args.datasets:
        source = args.input.expanduser().resolve() if args.input else root / f"{dataset}.csv"
        if not args.input and not source.is_file():
            source = root / "forecasting" / f"{dataset}.csv"
        if not source.is_file():
            parser.error(f"Missing dataset: {source}")
        dest = args.output.expanduser().resolve() / dataset
        if dest == source.parent or source.parent in dest.parents:
            parser.error("Output must be outside the source data directory")
        if dest.exists() and not args.overwrite:
            parser.error(f"Output exists: {dest}; choose a new output or use --overwrite")
        jobs.append((dataset, source, dest))
    for dataset, source, dest in jobs:
        run_dataset(dataset, source, dest, args.batch_size, args.overwrite)


if __name__ == "__main__":
    main()
