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
