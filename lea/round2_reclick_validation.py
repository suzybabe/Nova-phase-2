#!/usr/bin/env python3
"""Offline Round 2 reclick challenger. Run manually on the Huawei VM.

Reads only the provided Phase 2 data directory. Never uses test labels in model fit.
Historical slate-ranking evaluation only, NOT the official simulator score.
"""
import argparse
import numpy as np
import pandas as pd
from validation import load, rolling_splits, compute_metrics, SmoothedCTR

DAY = 86400.0


def score_reclick(pairs, ctr, hist, hours, strategy):
    """Score displayed candidates using TRAINING history only."""
    base = ctr.score(pairs)
    keys = pd.MultiIndex.from_frame(pairs[['user_id', 'itemID']])
    h = hist.reindex(keys)
    last = h['last_ts'].to_numpy(dtype=float)
    count = h['n_clicks'].to_numpy(dtype=float)
    age = (pairs['timestamp'].to_numpy(dtype=float) - last) / DAY
    eligible = np.isfinite(age) & (age >= 0)
    if hours is not None:
        eligible &= age <= hours / 24.0

    result = base.copy()
    if strategy == 'recent':
        # All eligible past-clicked items beat non-clicked items; most recent wins.
        result[eligible] = 2.0 + 1.0 / (1.0 + age[eligible]) + 0.0001 * base[eligible]
    elif strategy == 'frequency':
        # Frequency-first, with recency as a tie-breaker.
        result[eligible] = (2.0 + np.log1p(count[eligible])
                            + 0.05 / (1.0 + age[eligible])
                            + 0.0001 * base[eligible])
    else:
        raise ValueError(strategy)
    return result, eligible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('--days', type=float, default=1)
    ap.add_argument('--folds', type=int, default=2)
    args = ap.parse_args()

    feedback, catalog = load(args.data_dir)
    print('Round 2 historical-slate experiment (not official simulator score)')
    print('No test feedback is used to fit scores; catalog future validity is not used')
    experiments = [
        ('CTR 7d/prior20', None, None, 7.0, 20.0),
        ('CTR 1d/prior100', None, None, 1.0, 100.0),
        ('Reclick 12h', 12, 'recent', 1.0, 100.0),
        ('Reclick 72h', 72, 'recent', 1.0, 100.0),
        ('Reclick unlimited', None, 'recent', 1.0, 100.0),
        ('Frequency unlimited', None, 'frequency', 1.0, 100.0),
    ]
    summary = []
    for fold, train, test, cutoff in rolling_splits(feedback, args.days, args.folds):
        clicked_events = test.groupby('event_id')['clicked'].transform('sum') == 1
        slate = test.loc[clicked_events].reset_index(drop=True)
        if slate.empty:
            continue
        pairs = slate[['user_id', 'timestamp', 'itemID']]
        history = (train.loc[train['clicked'].eq(1)]
                   .groupby(['user_id', 'itemID'])['timestamp']
                   .agg(last_ts='max', n_clicks='size'))
        ctr_cache = {}
        print(f'\nFold {fold}: {slate.event_id.nunique():,} clicked slates; '
              f'{len(history):,} historical clicked user-item pairs')
        for name, hours, strategy, half_life, prior in experiments:
            key = (half_life, prior)
            if key not in ctr_cache:
                model = SmoothedCTR(half_life_days=half_life, prior_strength=prior)
                model.fit(train, catalog)
                ctr_cache[key] = model
            ctr = ctr_cache[key]
            if strategy is None:
                scores = ctr.score(pairs)
                eligible = np.zeros(len(pairs), dtype=bool)
            else:
                scores, eligible = score_reclick(pairs, ctr, history, hours, strategy)
            result = compute_metrics(slate, scores, set(train['user_id'].unique()))
            covered = (pd.Series(eligible).groupby(slate['event_id']).any().mean())
            clicked_repeat = int((eligible & slate['clicked'].to_numpy(dtype=bool)).sum())
            print(f'  {name:23s} hit@1={result["hit1"]:.4f} '
                  f'MRR={result["mrr"]:.4f} '
                  f'eligible_slates={covered:.1%} '
                  f'clicked_item_eligible={clicked_repeat:,}')
            summary.append(dict(fold=fold, name=name, n=result['n_events'],
                                hit1=result['hit1'], mrr=result['mrr'],
                                eligible_slates=covered))
    if not summary:
        raise RuntimeError('No evaluable folds')
    df = pd.DataFrame(summary)
    print('\nEvent-weighted summary:')
    for name, g in df.groupby('name', sort=False):
        w = g['n'].to_numpy()
        print(f'  {name:23s} hit@1={np.average(g.hit1, weights=w):.4f} '
              f'MRR={np.average(g.mrr, weights=w):.4f} '
              f'eligible_slates={np.average(g.eligible_slates, weights=w):.1%}')
    print('\nCAUTION: These are observed-slate comparisons. Off-slate recommendations '
          'cannot be evaluated from these labels. Confirm on a separate holdout.')


if __name__ == '__main__':
    main()
