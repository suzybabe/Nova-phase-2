"""NOVA LightGBM slate ranker (experimental, validation-first).

Historical catalogue intervals are NOT historical availability labels.
`score` returns finite ranking scores by default; the caller MUST filter
actual prediction candidates using catalog valid_from <= ts < valid_to.
For production, optionally set mask_availability=True when constructing the
ranker, or (preferably) prefilter candidates before calling score.

Only clicked slates train LambdaRank. All events update historical features.
Events sharing a timestamp see the same strictly earlier history.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb

DAY = 86400.0


class NovaRanker:
    FEATURE_NAMES = [
        'ctr_smoothed', 'log_impressions', 'log_clicks',
        'first_seen_age_days', 'log_user_slates', 'als_score', 'als_missing',
    ]

    def __init__(self, half_life_days=1.0, prior=100.0, n_estimators=250,
                 learning_rate=0.04, num_leaves=15, random_state=42,
                 mask_availability=False):
        if half_life_days <= 0 or prior <= 0:
            raise ValueError('half_life_days and prior must be positive')
        self.half_life_seconds = float(half_life_days) * DAY
        self.prior = float(prior)
        self.mask_availability = bool(mask_availability)
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
        required = {'itemID', 'valid_from', 'valid_to'}
        if not required.issubset(catalog.columns):
            raise ValueError(f'catalog missing {required - set(catalog.columns)}')
        if catalog.itemID.duplicated().any():
            raise ValueError('catalog has duplicate itemID')
        self._catalog = catalog.set_index('itemID')[['valid_from', 'valid_to']].copy()
        self._items = {}  # item -> (decayed impressions, clicks, latest timestamp, first seen)
        self._user_slates = defaultdict(int)
        self._last_ts = self._global_ts = None
        self._global_clicks = self._global_impressions = 0.0

    def _decay(self, value, delta):
        return value * np.exp(-np.log(2.0) * np.maximum(0.0, delta) / self.half_life_seconds)

    def _features_for(self, frame):
        """Vectorised scoring features; preserves input order and repeats."""
        if frame.empty:
            return np.empty((0, len(self.FEATURE_NAMES)), np.float32)
        items = frame['itemID'].astype(str)
        ts = frame['timestamp'].to_numpy(dtype=np.float64)
        users = frame['user_id'].astype(str)
        # Unique item lookup once per batch rather than one Python loop per pair.
        unique = pd.Index(items.unique())
        state = pd.DataFrame.from_dict(
            {k: self._items[k] for k in unique if k in self._items},
            orient='index', columns=['imp', 'clk', 'last', 'first'],
        ).reindex(items.to_numpy())
        imp0 = state['imp'].fillna(0).to_numpy(dtype=float)
        clk0 = state['clk'].fillna(0).to_numpy(dtype=float)
        last = np.where(state['last'].isna().to_numpy(), ts, state['last'].fillna(0).to_numpy(dtype=float))
        first = np.where(state['first'].isna().to_numpy(), ts, state['first'].fillna(0).to_numpy(dtype=float))
        decay = self._decay(1.0, ts - last)
        imp, clk = imp0 * decay, clk0 * decay
        if self._global_ts is None or self._global_impressions <= 0:
            prior_mean = np.full(len(frame), 0.5)
        else:
            global_decay = self._decay(1.0, ts - self._global_ts)
            prior_mean = (self._global_clicks * global_decay + 1.0) / (
                self._global_impressions * global_decay + 2.0)
        ctr = (clk + self.prior * prior_mean) / (imp + self.prior)
        fatigue = users.map(self._user_slates).fillna(0).to_numpy(dtype=float)
        if 'als_score' in frame:
            als = pd.to_numeric(frame['als_score'], errors='coerce').to_numpy(dtype=float)
        else:
            als = np.full(len(frame), np.nan)
        missing = ~np.isfinite(als)
        X = np.column_stack((
            ctr, np.log1p(imp), np.log1p(clk),
            np.maximum(0, (ts - first) / DAY), np.log1p(fatigue),
            np.where(missing, 0, als), missing.astype(float),
        ))
        return X.astype(np.float32)

    def _update_event(self, event):
        ts = float(event['timestamp'].iloc[0])
        if self._last_ts is not None and ts < self._last_ts:
            raise ValueError('History updates must be chronological')
        self._last_ts = ts
        self._user_slates[str(event['user_id'].iloc[0])] += 1
        if self._global_ts is not None:
            factor = self._decay(1.0, ts - self._global_ts)
            self._global_clicks *= factor
            self._global_impressions *= factor
        self._global_ts = ts
        self._global_clicks += float(event.clicked.sum())
        self._global_impressions += len(event)
        for row in event.itertuples(index=False):
            item = str(row.itemID)
            past = self._items.get(item)
            if past is None:
                self._items[item] = (1., float(row.clicked), ts, ts)
            else:
                factor = self._decay(1.0, ts - past[2])
                self._items[item] = (
                    past[0] * factor + 1, past[1] * factor + float(row.clicked),
                    ts, past[3],
                )

    def fit(self, train, catalog):
        required = {'event_id', 'user_id', 'timestamp', 'itemID', 'clicked'}
        if not required.issubset(train.columns):
            raise ValueError(f'train missing {required - set(train.columns)}')
        self._reset(catalog)
        data = train.copy()
        data['itemID'] = data['itemID'].astype(str)
        data['user_id'] = data['user_id'].astype(str)
        data = data.sort_values(['timestamp', 'event_id'], kind='stable')
        if (data.groupby('event_id')['timestamp'].nunique() > 1).any() or \
           (data.groupby('event_id')['user_id'].nunique() > 1).any():
            raise ValueError('Each event must have one timestamp and user')
        features, labels, groups = [], [], []
        for _, same_time in data.groupby('timestamp', sort=False):
            # Critical: do not update history until ALL same-timestamp
            # events have had features computed.
            for _, event in same_time.groupby('event_id', sort=False):
                if event.clicked.sum() == 1 and len(event) > 1:
                    features.append(self._features_for(event))
                    labels.append(event.clicked.to_numpy(dtype=np.int32))
                    groups.append(len(event))
            for _, event in same_time.groupby('event_id', sort=False):
                self._update_event(event)
        if not groups:
            raise ValueError('No clicked events with at least 2 candidates')
        X = np.vstack(features)
        y = np.concatenate(labels)
        self.model = lgb.LGBMRanker(**self.model_params)
        self.model.fit(X, y, group=groups, feature_name=self.FEATURE_NAMES)
        return self

    def score(self, pairs):
        """Return one score for each input pair, in the original order.

        Default: finite scores for historical slate validation, ignoring current
        prediction-window availability. Production: prefilter valid candidates
        or construct NovaRanker(mask_availability=True).
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
        scores = np.asarray(self.model.predict(self._features_for(pairs)), dtype=float)
        if self.mask_availability:
            cat = self._catalog.reindex(pairs['itemID'].astype(str).to_numpy())
            ts = pairs['timestamp'].to_numpy(dtype=float)
            available = ((cat['valid_from'].to_numpy(dtype=float) <= ts) &
                         (ts < cat['valid_to'].to_numpy(dtype=float)))
            scores[~available] = -np.inf
        return scores
