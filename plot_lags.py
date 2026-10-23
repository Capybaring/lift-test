#!/usr/bin/env python3
"""Plot dataset circular LIFT lags in exact one-hour bins."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

COLORS = ["#2563EB", "#7C3AED"]
HISTORY, HORIZON = 168, 24


def prepare_distribution(frame, exogenous):
    required = {"window_id", "exogenous", "endogenous", "lag_steps"}
    if not required.issubset(frame):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if set(frame.endogenous) != {"OT"} or set(frame.exogenous) != set(exogenous):
        raise ValueError("Expected both requested exogenous -> OT pairs")
    if frame.duplicated(["window_id", "exogenous", "endogenous"]).any():
        raise ValueError("Duplicate window/pair rows")
    if frame.groupby("window_id").size().ne(2).any():
        raise ValueError("Each window must contain both pairs")
    lag = pd.to_numeric(frame.lag_steps, errors="raise").to_numpy(dtype=float)
    if not (np.isfinite(lag).all() and (lag == np.floor(lag)).all()
            and ((lag >= 1) & (lag <= HISTORY - 2)).all()):
        raise ValueError("Expected integer lags in 1..166 hours")
    tables, stats = [], []
    grid = np.arange(1, HISTORY - 1)
    for name in exogenous:
        values = frame.loc[frame.exogenous == name, "lag_steps"].to_numpy(dtype=int)
        counts = np.bincount(values, minlength=HISTORY)[grid]
        tables.append(pd.DataFrame(dict(exogenous=name, lag_hours=grid,
            windows=counts, fraction=counts / len(values), percent=100 * counts / len(values))))
        top = np.argsort(-counts, kind="stable")[:3]
        stats.append(dict(exogenous=name, windows=len(values),
            mean_lag_hours=float(values.mean()), median_lag_hours=float(np.median(values)),
            top_lags=[dict(lag_hours=int(grid[i]), percent=float(100 * counts[i] / len(values)))
                      for i in top if counts[i] > 0]))
    return pd.concat(tables, ignore_index=True), stats


def plot_distributions(frame, output, dataset, exogenous, stride_hours=168):
    starts = np.sort(frame.window_id.unique().astype(int))
    if len(starts) > 1 and not (np.diff(starts) == stride_hours).all():
        raise ValueError("Window spacing does not match stride-hours")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "be_lift_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, MultipleLocator

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table, stats = prepare_distribution(frame, exogenous)
    table.to_csv(output / f"{dataset}_lag_distribution_counts.csv", index=False)
    info = dict(dataset=dataset, history_hours=HISTORY, forecast_hours=HORIZON, stride_hours=stride_hours,
                bin_width_hours=1, estimator="circular FFT with official local-peak selection",
                pairs=stats)
    (output / f"{dataset}_lag_distribution_stats.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths = []
    with plt.rc_context({"font.family":"DejaVu Sans", "font.size":10,
                         "text.color":"#0F172A", "svg.fonttype":"none"}):
        fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.4))
        fig.patch.set_facecolor("#F8FAFC")
        ymax = min(100., max(10., np.ceil(table.percent.max() * 1.5 / 10) * 10))
        for ax, name, color, item in zip(axes, exogenous, COLORS, stats):
            selected = table[table.exogenous == name]
            ax.bar(selected.lag_hours, selected.percent, width=.86, color=color, zorder=3)
            ax.axvline(item["mean_lag_hours"], color="#D97706", linestyle="--", linewidth=1.8,
                       label=f"Mean {item['mean_lag_hours']:.2f} h")
            ax.set(xlim=(0, HISTORY), ylim=(0, ymax), xlabel="Selected circular lag (hours)",
                   ylabel="Share of sampled windows")
            ax.xaxis.set_major_locator(MultipleLocator(24))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            ax.grid(axis="y", color="#E2E8F0", zorder=0)
            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)
            ax.set_title(f"{name.strip()}  →  OT", loc="left", fontweight="bold", pad=12)
            ax.text(.015, .94, f"N = {item['windows']:,} sampled windows", transform=ax.transAxes,
                    va="top", fontsize=9, color="#475569")
            top_text = "  |  ".join(f"{v['lag_hours']} h: {v['percent']:.2f}%" for v in item["top_lags"])
            ax.text(.015, .85, f"Most common exact lags: {top_text}", transform=ax.transAxes,
                    va="top", fontsize=9, color="#475569")
            ax.legend(loc="upper right", frameon=False, fontsize=9)
        fig.suptitle(f"{dataset} · Distribution of LIFT-selected circular lags", x=.07, y=.98,
                     ha="left", fontsize=17, fontweight="bold")
        fig.text(.07, .934,
                 f"History: 168 h   |   Forecast horizon: 24 h   |   Sliding stride: {stride_hours} h   |   Bin width: 1 h",
                 color="#475569")
        fig.text(.07, .035, "Circular alignment within each history window. Weekly sampling does not remove wraparound.",
                 fontsize=9, color="#475569")
        fig.subplots_adjust(left=.075, right=.975, bottom=.12, top=.86, hspace=.52)
        for suffix in ("png", "svg"):
            path = output / f"{dataset}_lag_distribution_hours.{suffix}"
            fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
            paths.append(str(path.resolve()))
        plt.close(fig)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=str.upper, choices=["BE", "NP", "DE", "PJM", "FR"])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stride-hours", type=int, default=168,
                        help="Validate actual window spacing; does not subsample existing results")
    args = parser.parse_args(argv)
    source = args.results.expanduser().resolve()
    if source.is_dir():
        source = source / f"{args.dataset}_lift_window_lags.csv.gz"
    from lift_weekly import DATASETS
    paths = plot_distributions(pd.read_csv(source), args.output or source.parent / "plots",
                               args.dataset, DATASETS[args.dataset], args.stride_hours)
    print("\n".join(paths))


if __name__ == "__main__":
    main()
