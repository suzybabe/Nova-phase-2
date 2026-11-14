"""Past-only sequence diagnostics for NOVA Phase 2. Run on authorised Huawei VM.

Usage: python sequence_audit.py --data-dir ~/data/round-2 --output sequence_report.json
No model fitting or competition data processing is performed by the assistant.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86400.0


def rate(n, d):
    return round(float(n) / float(d), 6) if d else None


def age_bucket(days):
    if days is None:
        return 'never'
    if days < .5:
        return '<12h'
    if days < 1:
        return '12-24h'
    if days < 3:
        return '1-3d'
    if days < 8:
        return '3-8d'
    return '8d+'


def exposure_bucket(n):
    return str(n) if n <= 4 else ('5-9' if n < 10 else '10+')


def summarize(counter):
    return {k: {'shown': v[0], 'clicked': v[1], 'observed_ctr': rate(v[1], v[0])}
            for k, v in sorted(counter.items())}


def audit(data, min_history_timestamp=None):
    required = {'event_id', 'user_id', 'timestamp', 'itemID', 'clicked'}
    if not required.issubset(data.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(data.columns))}')
    data = data.copy()
    data['user_id'] = data['user_id'].astype(str)
    data['itemID'] = data['itemID'].astype(str)
    data['timestamp'] = pd.to_numeric(data['timestamp'], errors='raise')
    data['clicked'] = pd.to_numeric(data['clicked'], errors='raise').astype(int)
    if not data.clicked.isin([0, 1]).all():
        raise ValueError('clicked must be binary')
    if data.duplicated(['event_id', 'itemID']).any():
        raise ValueError('duplicate event-item pair')
    if (data.groupby('event_id').user_id.nunique() > 1).any() or (data.groupby('event_id').timestamp.nunique() > 1).any():
        raise ValueError('event contains multiple users or timestamps')
    data = data.sort_values(['timestamp', 'event_id'], kind='stable')

    # state: past clicks, past exposures, most recent click and timestamp, per user
    clicks = defaultdict(Counter)
    last_click = defaultdict(dict)
    impressions = defaultdict(Counter)
    by_age = defaultdict(lambda: [0, 0])
    by_exposure = defaultdict(lambda: [0, 0])
    by_repeat = defaultdict(lambda: [0, 0])
    clicked_slates = 0
    recent_in_slate = 0
    frequent_in_slate = 0
    both_in_slate = 0
    recent_wins = 0
    frequent_wins = 0
    tie = 0
    eval_events = 0
    eval_rows = 0
    prev_clicked_in_slate = 0
    history_users = set()
    # Same-timestamp events must not leak into each other. Stage updates until timestamp changes.
    staged = []
    current_ts = None

    def flush():
        for user, item, ts, clicked in staged:
            impressions[user][item] += 1
            if clicked:
                clicks[user][item] += 1
                last_click[user][item] = ts
            history_users.add(user)
        staged.clear()

    for event_id, group in data.groupby('event_id', sort=False):
        ts = float(group.timestamp.iloc[0]); user = group.user_id.iloc[0]
        if current_ts is not None and ts != current_ts:
            flush()
        current_ts = ts
        items = group.itemID.tolist()
        labels = group.clicked.tolist()
        eligible = min_history_timestamp is None or ts >= min_history_timestamp
        if eligible:
            eval_events += 1
            eval_rows += len(items)
            clicked_slates += int(sum(labels) == 1)
            # No assumption that shown items were chosen randomly.
            for item, label in zip(items, labels):
                n = impressions[user][item]
                last = last_click[user].get(item)
                age = None if last is None else max(0., (ts - last) / DAY)
                by_age[age_bucket(age)][label] += 1
                by_exposure[exposure_bucket(n)][label] += 1
                by_repeat['clicked_before' if clicks[user][item] else 'not_clicked_before'][label] += 1
                prev_clicked_in_slate += int(n > 0 and clicks[user][item] > 0)
            if sum(labels) == 1:
                shown_prior_clicks = [i for i in items if clicks[user][i] > 0]
                if shown_prior_clicks:
                    recent_item = max(shown_prior_clicks, key=lambda i: (last_click[user][i], i))
                    frequent_item = max(shown_prior_clicks, key=lambda i: (clicks[user][i], last_click[user][i], i))
                    recent_in_slate += 1
                    frequent_in_slate += 1
                    both_in_slate += 1
                    selected = items[labels.index(1)]
                    if recent_item == frequent_item:
                        tie += 1
                    else:
                        recent_wins += int(selected == recent_item)
                        frequent_wins += int(selected == frequent_item)
        staged.extend((user, item, ts, int(label)) for item, label in zip(items, labels))
    flush()

    # Accumulator indexed by click label: [0] = unclicked, [1] = clicked.
    def convert(d):
        return {k: {'shown': int(v[0] + v[1]), 'clicked': int(v[1]),
                    'observed_ctr': rate(v[1], v[0] + v[1])} for k, v in sorted(d.items())}

    return {
        'warning': 'Observed shown-slate associations only; not causal policy value or official score.',
        'evaluated_events': eval_events, 'evaluated_impression_rows': eval_rows,
        'clicked_slates': clicked_slates,
        'observed_by_last_click_age': convert(by_age),
        'observed_by_prior_exposures': convert(by_exposure),
        'observed_by_prior_click': convert(by_repeat),
        'prior_clicked_candidate_impressions': prev_clicked_in_slate,
        'clicked_slates_with_prior_clicked_candidate': recent_in_slate,
        'recent_vs_frequent_when_both_shown': {
            'slates_with_at_least_one_previously_clicked_item': both_in_slate,
            'same_recommendation_count': tie,
            'different_recommendation_count': both_in_slate - tie,
            'recent_item_was_clicked_when_different': recent_wins,
            'frequent_item_was_clicked_when_different': frequent_wins,
            'neither_was_clicked_when_different': both_in_slate - tie - recent_wins - frequent_wins,
        },
        'distinct_users_with_history': len(history_users),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True, type=Path)
    p.add_argument('--output', default='sequence_report.json', type=Path)
    p.add_argument('--evaluation-start', type=float, default=None,
                   help='Unix timestamp: evaluate only events from this time; earlier events form history')
    args = p.parse_args()
    parquet = args.data_dir / 'feedback_long.parquet'
    csv = args.data_dir / 'feedback_long.csv'
    if parquet.exists():
        frame = pd.read_parquet(parquet, columns=['event_id', 'user_id', 'timestamp', 'itemID', 'clicked'])
    elif csv.exists():
        frame = pd.read_csv(csv, usecols=['event_id', 'user_id', 'timestamp', 'itemID', 'clicked'])
    else:
        raise FileNotFoundError('No feedback_long.parquet or feedback_long.csv in data directory')
    report = audit(frame, args.evaluation_start)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
