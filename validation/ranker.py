"""Leakage-aware LightGBM slate ranker for Tech Arena Phase 2.

Input train: long-form impression rows with event_id,user_id,timestamp,itemID,clicked.
Catalog: itemID,valid_from,valid_to.
Input pairs: candidate impression rows with user_id,timestamp,itemID and optional
als_score. Output: numpy scores in input order; unavailable items score -inf.

Train only clicked events (exactly one positive per event). Training features
are generated prequentially: history is updated *after* scoring each event.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb


class NovaRanker:
    FEATURE_NAMES = [
        'ctr_smoothed', 'log_impressions', 'log_clicks', 'age_days',
        'time_left_days', 'log_user_slates', 'als_score', 'als_missing',
    ]

    def __init__(self, half_life_days=1.0, prior=100.0, n_estimators=250,
                 learning_rate=0.04, num_leaves=15, random_state=42):
        self.half_life_seconds = float(half_life_days) * 86400.0
        self.prior = float(prior)
        self.model_params = dict(objective='lambdarank', metric='ndcg',
                                 n_estimators=n_estimators,
                                 learning_rate=learning_rate,
                                 num_leaves=num_leaves, random_state=random_state,
                                 verbosity=-1, n_jobs=8)
        self.model = None
        self._catalog = None
        self._items = {}
        self._user_slates = defaultdict(int)
        self._last_ts = None
        self._global_clicks = 0.0
        self._global_impressions = 0.0
        self._global_ts = None

    def _reset(self, catalog):
        cat = catalog.copy()
        required = {'itemID', 'valid_from', 'valid_to'}
        if not required.issubset(cat.columns):
            raise ValueError(f'catalog missing {required - set(cat.columns)}')
        if cat.itemID.duplicated().any():
            raise ValueError('catalog has duplicate itemID')
        self._catalog = cat.set_index('itemID')[['valid_from', 'valid_to']]
        self._items = {}  # itemID -> [decayed impressions, decayed clicks, last_ts]
        self._user_slates = defaultdict(int)
        self._last_ts = None
        self._global_clicks = 0.0
        self._global_impressions = 0.0
        self._global_ts = None

    def _decay(self, value, dt):
        return value * np.exp(-np.log(2.0) * max(0.0, dt) / self.half_life_seconds)

    def _global_prior(self, ts):
        if self._global_impressions <= 0:
            return 0.5
        factor = self._decay(1.0, ts - self._global_ts)
        return (self._global_clicks * factor + 1.0) / (self._global_impressions * factor + 2.0)

    def _features_for(self, frame):
        rows = []
        for r in frame.itertuples(index=False):
            item, user, ts = str(r.itemID), str(r.user_id), float(r.timestamp)
            state = self._items.get(item)
            if state is None:
                imp = clk = 0.0
            else:
                factor = self._decay(1.0, ts - state[2])
                imp, clk = state[0] * factor, state[1] * factor
            ctr = (clk + self.prior * self._global_prior(ts)) / (imp + self.prior)
            if item in self._catalog.index:
                valid_from, valid_to = self._catalog.loc[item]
                age = max(0.0, (ts - float(valid_from)) / 86400.0)
                left = max(0.0, (float(valid_to) - ts) / 86400.0)
                available = float(valid_from) <= ts < float(valid_to)
            else:
                age, left, available = 0.0, 0.0, False
            als = getattr(r, 'als_score', np.nan)
            missing = pd.isna(als)
            rows.append([ctr, np.log1p(imp), np.log1p(clk), age, left,
                         np.log1p(self._user_slates[user]),
                         0.0 if missing else float(als), float(missing)])
        return np.asarray(rows, dtype=np.float32), np.asarray([
            str(r.itemID) in self._catalog.index and
            float(self._catalog.loc[str(r.itemID), 'valid_from']) <= float(r.timestamp) <
            float(self._catalog.loc[str(r.itemID), 'valid_to'])
            for r in frame.itertuples(index=False)], dtype=bool)

    def _update_event(self, event):
        ts = float(event.timestamp.iloc[0])
        if self._last_ts is not None and ts < self._last_ts:
            raise ValueError('History updates must be chronological')
        self._last_ts = ts
        user = str(event.user_id.iloc[0])
        self._user_slates[user] += 1
        if self._global_ts is not None:
            f = self._decay(1.0, ts - self._global_ts)
            self._global_clicks *= f
            self._global_impressions *= f
        self._global_ts = ts
        self._global_clicks += float(event.clicked.sum())
        self._global_impressions += len(event)
        for r in event.itertuples(index=False):
            key = str(r.itemID)
            state = self._items.get(key)
            if state is None:
                imp = clk = 0.0
            else:
                f = self._decay(1.0, ts - state[2])
                imp, clk = state[0] * f, state[1] * f
            self._items[key] = [imp + 1.0, clk + float(r.clicked), ts]

    def fit(self, train, catalog):
        """Fit using past-only features; retains complete history for later scoring.

        Rows from an event must share a timestamp and user_id. Train input must
        contain all impressions, including non-clicked events, for correct history.
        """
        required = {'event_id', 'user_id', 'timestamp', 'itemID', 'clicked'}
        if not required.issubset(train.columns):
            raise ValueError(f'train missing {required - set(train.columns)}')
        self._reset(catalog)
        data = train.copy()
        data['itemID'] = data['itemID'].astype(str)
        data['user_id'] = data['user_id'].astype(str)
        data = data.sort_values(['timestamp', 'event_id'], kind='stable')
        features, labels, groups = [], [], []
        # Sort by event timestamp, and never update within a slate.
        for _, event in data.groupby('event_id', sort=False):
            if event.timestamp.nunique() != 1 or event.user_id.nunique() != 1:
                raise ValueError('Each event must have one timestamp and user')
            if event.clicked.sum() == 1 and len(event) > 1:
                x, available = self._features_for(event)
                # Keep the slate intact, but exclude impossible candidates by
                # masking them in inference rather than altering group sizes.
                features.append(x)
                labels.append(event.clicked.to_numpy(dtype=np.int32))
                groups.append(len(event))
            self._update_event(event)
        if not groups:
            raise ValueError('No clicked events with at least 2 candidates')
        X = np.vstack(features)
        y = np.concatenate(labels)
        self.model = lgb.LGBMRanker(**self.model_params)
        self.model.fit(X, y, group=groups, feature_name=self.FEATURE_NAMES)
        return self

    def score(self, pairs):
        """Score candidate pairs in original row order; unavailable items get -inf.

        pairs must be at or after the latest training timestamp. If scoring a
        future round, refit on its cumulative feedback to update the history.
        """
        if self.model is None:
            raise RuntimeError('Call fit(train, catalog) first')
        required = {'user_id', 'timestamp', 'itemID'}
        if not required.issubset(pairs.columns):
            raise ValueError(f'pairs missing {required - set(pairs.columns)}')
        if len(pairs) == 0:
            return np.empty(0, dtype=np.float64)
        if self._last_ts is not None and float(pairs.timestamp.min()) < self._last_ts:
            raise ValueError('Scoring timestamps precede fitted history; temporal leakage risk')
        X, available = self._features_for(pairs)
        scores = self.model.predict(X).astype(np.float64)
        scores[~available] = -np.inf
        return scores
