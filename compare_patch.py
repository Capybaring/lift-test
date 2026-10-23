#!/usr/bin/env python3
"""Paired raw-hour / scalar patch-mean circular LIFT diagnostics; no TTCN."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch

from lift_weekly import DATASETS, L, H, STRIDE, UPSTREAM_COMMIT, read_dataset, lead_estimator, write_json

BASE_MODEL_COMMIT = '352c88613ee0f4d9597591b2e49ccce4a490e2ae'


def patch_intervals(length, width, stride):
    """Same half-open physical-time bins as exobase.data.pack_patches.

    Hourly complete data makes integer time boundaries equal to array indices.
    The final bin is clipped, never padded with a made-up observation.
    """
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (length, width, stride)):
        raise ValueError('Patch lengths and strides must be integer hours')
    if not 0 < stride <= width <= length:
        raise ValueError('Require 0 < patch stride <= patch width <= history')
    count = math.ceil(max(length-width, 0)/stride)+1
    return [(p*stride, min(p*stride+width, length)) for p in range(count)]


def patch_means(history, width, stride=None):
    """[B,C,168] -> [B,C,K]; per-bin observed scalar mean, no neural encoder."""
    stride = width if stride is None else stride
    intervals = patch_intervals(history.shape[-1], width, stride)
    return torch.stack([history[..., a:b].mean(-1) for a, b in intervals], -1)


def paired_estimates(history, width, stride):
    raw_lag, raw_score, _ = lead_estimator(history)
    patches = patch_means(history, width, stride)
    patch_lag, patch_score, _ = lead_estimator(patches)
    return raw_lag, raw_score, patch_lag, patch_score


def plot_comparison(frame, dataset, destination, width, stride, window_stride, token_count):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'lift_compare_mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, MultipleLocator
    destination.mkdir(parents=True, exist_ok=True)
    grid = np.arange(1, L)
    tables, summaries = [], []
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)
    for row, exo in enumerate(DATASETS[dataset]):
        pair = frame[frame.exogenous == exo]
        for col, mode in enumerate(('raw', 'patch')):
            values = pair[f'{mode}_lag_hours'].to_numpy(int)
            counts = np.bincount(values, minlength=L)[grid]
            percent = counts*100/len(values)
            tables.append(pd.DataFrame(dict(dataset=dataset, exogenous=exo, mode=mode,
                lag_hours=grid, windows=counts, percent=percent)))
            top = np.argsort(-counts, kind='stable')[:3]
            summaries.append(dict(exogenous=exo, mode=mode, windows=len(values),
                mean_lag_hours=float(values.mean()), mean_abs_score=float(pair[f'{mode}_score'].abs().mean()),
                zero_score_fallback_windows=int(pair[f'{mode}_zero_score_fallback'].sum()),
                most_common=[dict(hours=int(grid[i]), windows=int(counts[i])) for i in top if counts[i]]))
            ax = axes[row, col]
            ax.bar(grid, percent, width=.9, color=('#2563EB' if col == 0 else '#7C3AED'))
            ax.axvline(values.mean(), color='#D97706', linestyle='--', label=f'Mean {values.mean():.2f} h')
            ax.set_title(f'{exo.strip()} → OT\n'+('Hourly values' if col == 0 else f'{width} h patch means'))
            ax.set(xlim=(0, L), ylim=(0, 100), xlabel='Selected circular lag (hours)', ylabel='Share of sampled windows')
            ax.xaxis.set_major_locator(MultipleLocator(24))
            ax.yaxis.set_major_formatter(PercentFormatter(100))
            ax.grid(axis='y', alpha=.2)
            ax.legend(fontsize=9)
            ax.text(.02, .9, ' | '.join(f'{grid[i]} h: {percent[i]:.1f}%' for i in top if counts[i]),
                    transform=ax.transAxes, fontsize=9)
    fig.suptitle(f'{dataset} · Paired circular LIFT comparison · {frame.window_id.nunique()} windows', fontsize=16)
    fig.text(.5, .92, f'History {L} h | Horizon {H} h | Window stride {window_stride} h | Patch width/stride {width}/{stride} h', ha='center')
    fig.text(.5, .025, f'Raw candidates: 1–166 h. Patch candidates: 1–{token_count-2} shifts = {stride}–{(token_count-2)*stride} h. Each bar counts one exact hour.', ha='center', fontsize=10)
    fig.subplots_adjust(top=.82, bottom=.13, hspace=.5, wspace=.22)
    for ext in ('png', 'svg'):
        fig.savefig(destination/f'{dataset}_raw_vs_patch.{ext}', dpi=160, bbox_inches='tight')
    plt.close(fig)
    pd.concat(tables).to_csv(destination/f'{dataset}_distribution_hours.csv', index=False)
    return summaries


def run_comparison(dataset, source, output, width=12, patch_stride=12, window_stride=168, batch_size=512):
    times, data = read_dataset(source, DATASETS[dataset])
    starts = np.arange(0, len(times)-L-H+1, window_stride)
    intervals = patch_intervals(L, width, patch_stride)
    if len(intervals) < 3 or width > H:
        raise ValueError('Need at least 3 historical patches and width <= forecast horizon (base collator rule)')
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(status='running', dataset=dataset, history_hours=L, forecast_hours=H,
        window_stride_hours=window_stride, patch_width_hours=width, patch_stride_hours=patch_stride,
        history_patch_count=len(intervals), patch_intervals_hours=intervals, patch_reduction='observed arithmetic mean',
        raw_candidates_hours=[1,L-2], patch_candidates_steps=[1,len(intervals)-2],
        patch_candidates_hours=[patch_stride,(len(intervals)-2)*patch_stride],
        official_commit=UPSTREAM_COMMIT, base_model_commit=BASE_MODEL_COMMIT,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        preprocessing='raw float32 -> patch means (patch mode only) -> window instance_norm epsilon=1e-8',
        direction='positive d: x[(k-d) mod sequence_length] aligned to y[k]',
        boundary='left-closed right-open; final patch clipped; anchored at each history start',
        fallback='Official lag=1 score=0 retained, flagged and included in distributions',
        scope='history only; no TTCN, train-split scaler, forecasting network or future values')
    write_json(output/'metadata.json', metadata)
    rows = []
    try:
        source_tensor = torch.as_tensor(data, dtype=torch.float32)
        for begin in range(0,len(starts),batch_size):
            ss = starts[begin:begin+batch_size]
            history = source_tensor[ss[:,None]+np.arange(L)].permute(0,2,1)
            rl, rs, pl, ps = [t.numpy() for t in paired_estimates(history,width,patch_stride)]
            rows.append(pd.DataFrame(dict(window_id=np.repeat(ss,2), dataset=dataset,
                history_start=np.repeat(times[ss],2), history_end=np.repeat(times[ss+L-1],2),
                forecast_start=np.repeat(times[ss+L],2), forecast_end=np.repeat(times[ss+L+H-1],2),
                exogenous=np.tile(DATASETS[dataset],len(ss)), endogenous='OT',
                raw_lag_hours=rl.ravel(), raw_score=rs.ravel(), raw_abs_score=np.abs(rs).ravel(),
                patch_lag_steps=pl.ravel(), patch_lag_hours=(pl*patch_stride).ravel(),
                patch_score=ps.ravel(), patch_abs_score=np.abs(ps).ravel(),
                raw_zero_score_fallback=(rs==0).ravel(), patch_zero_score_fallback=(ps==0).ravel())))
        frame = pd.concat(rows,ignore_index=True)
        frame.to_csv(output/f'{dataset}_raw_vs_patch.csv.gz',index=False,date_format='%Y-%m-%d %H:%M:%S',float_format='%.10g')
        summaries = plot_comparison(frame,dataset,output/'plots',width,patch_stride,window_stride,len(intervals))
        write_json(output/'summary.json', summaries)
        metadata.update(status='complete',windows=len(starts),rows=len(frame))
        write_json(output/'metadata.json',metadata)
        print(f'{dataset}: {len(starts)} paired windows, {len(frame)} rows -> {output}')
    except Exception as exc:
        metadata.update(status='failed',error=str(exc))
        write_json(output/'metadata.json',metadata)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',type=str.upper,choices=list(DATASETS),default=list(DATASETS))
    parser.add_argument('--data-dir',type=Path,default=Path.home()/'Downloads/dataset')
    parser.add_argument('--output',type=Path,default=Path('comparison_results'))
    parser.add_argument('--patch-hours',type=int,default=12)
    parser.add_argument('--patch-stride-hours',type=int,help='Default: same as patch width')
    parser.add_argument('--window-stride-hours',type=int,default=STRIDE)
    parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args(argv)
    stride=args.patch_hours if args.patch_stride_hours is None else args.patch_stride_hours
    try:
        intervals=patch_intervals(L,args.patch_hours,stride)
        if len(intervals)<3 or args.patch_hours>H:
            raise ValueError('Require >=3 historical patches and patch width <=24 hours')
        if args.window_stride_hours<1 or args.batch_size<1:
            raise ValueError('Window stride and batch size must be positive')
        if len(set(args.datasets))!=len(args.datasets):
            raise ValueError('Duplicate datasets')
        jobs=[]
        for ds in args.datasets:
            source=args.data_dir.expanduser().resolve()/f'{ds}.csv'
            if not source.is_file(): source=source.parent/'forecasting'/source.name
            if not source.is_file(): raise ValueError(f'Missing file: {source}')
            destination=args.output.expanduser().resolve()/ds
            if destination==source.parent or source.parent in destination.parents:
                raise ValueError('Output must be outside source data directory')
            if destination.exists() and not args.overwrite:
                raise ValueError(f'Output exists: {destination}; choose a new output or --overwrite')
            jobs.append((ds,source,destination))
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(1)
    for ds,source,destination in jobs:
        run_comparison(ds,source,destination,args.patch_hours,stride,args.window_stride_hours,args.batch_size)


if __name__=='__main__':
    main()
