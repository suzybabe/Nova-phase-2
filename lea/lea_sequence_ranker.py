#!/usr/bin/env python3
"""Lea's NOVA SequenceRanker: independent Phase-2 model challenger.

OFFLINE research only. Training/validation must be run manually on the designated
Huawei VM in accordance with the competition rules.

.fit(train,catalog) and .score(pairs) work with the team's validation toolkit.
Never treats test labels as training data. Historical-slate metrics are NOT the
official pooled-slate simulator result. Uses no Phase 1 artifacts or weights.

Features: global smoothed item CTR, user-item click frequency, recency-decayed
click count, and optional simple exposure response. Defaults keep exposure
weight zero until a properly validated ablation supports using it.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400.0


class LeaSequenceRanker:
    def __init__(self, click_half_life_days=3.0, click_weight=1.0,
                 recency_weight=1.0, ctr_weight=0.3,
                 exposure_weight=0.0, prior=100.0, eps=1e-9):
        self.click_half_life_days = float(click_half_life_days)
        self.click_weight = float(click_weight)
        self.recency_weight = float(recency_weight)
        self.ctr_weight = float(ctr_weight)
        self.exposure_weight = float(exposure_weight)
        self.prior = float(prior)
        self.eps = float(eps)
        if self.click_half_life_days <= 0 or self.prior <= 0:
            raise ValueError('half life and prior must be positive')
        self._ready = False

    def fit(self, train: pd.DataFrame, catalog: pd.DataFrame):
        required = {'user_id','itemID','timestamp','event_id','clicked'}
        if not required.issubset(train):
            raise ValueError(f'Missing training fields {required - set(train)}')
        if not {'itemID','valid_from','valid_to'}.issubset(catalog):
            raise ValueError('catalog needs itemID,valid_from,valid_to')
        # Only feedback in THIS training fold is used to build histories.
        tr = train[['user_id','itemID','timestamp','event_id','clicked']].copy()
        tr['clicked'] = pd.to_numeric(tr['clicked']).astype(int)
        tr['user_id'] = tr['user_id'].astype(str)
        tr['itemID'] = tr['itemID'].astype(str)
        tr['timestamp'] = pd.to_numeric(tr['timestamp']).astype('int64')
        if tr.empty:
            raise ValueError('Empty training data')
        self.train_end = int(tr.timestamp.max())
        self.global_ctr = (float(tr.clicked.sum()) + 1) / (len(tr) + 2)
        self.item_stats = tr.groupby('itemID', sort=False)['clicked'].agg(
            impressions='size', clicks='sum')
        self.clicks = (tr[tr.clicked.eq(1)]
                       .groupby(['user_id','itemID'], sort=False)['timestamp']
                       .agg(clicks='size',last_click='max'))
        # Keep individual click timestamps for exact recency weighting at query time.
        clicked = tr.loc[tr.clicked.eq(1), ['user_id','itemID','timestamp']]
        self.click_times = {
            (u,i): np.sort(v.to_numpy(dtype=np.int64))
            for (u,i),v in clicked.groupby(['user_id','itemID'])['timestamp']
        }
        # Impression count for an optional exposure-response component.
        self.exposures = (tr.groupby(['user_id','itemID'], sort=False)
                          .size().rename('n_exposures'))
        cat = catalog[['itemID','valid_from','valid_to']].copy()
        cat['itemID'] = cat.itemID.astype(str)
        if cat.itemID.duplicated().any():
            raise ValueError('Duplicate catalog item IDs')
        self.catalog = cat.set_index('itemID')
        self._ready = True
        return self

    def score(self, pairs: pd.DataFrame) -> np.ndarray:
        """Finite scores for logged candidates and full-catalog candidates.

        Availability is NOT masked here because the historical feedback and
        scoring-round catalog have different validity windows. Use
        choose_available() for legal full-catalog recommendations.
        """
        if not self._ready:
            raise RuntimeError('Call fit first')
        required = {'user_id','itemID','timestamp'}
        if not required.issubset(pairs):
            raise ValueError(f'Missing scoring fields {required - set(pairs)}')
        if pairs.empty:
            return np.empty(0, dtype=float)
        timestamps = pairs['timestamp'].to_numpy(dtype='int64')
        if timestamps.min() < self.train_end:
            raise ValueError('Scoring before end of training history would leak future interactions')
        keys = pd.MultiIndex.from_arrays([
            pairs.user_id.astype(str),pairs.itemID.astype(str)
        ], names=['user_id','itemID'])
        item_stats = self.item_stats.reindex(pairs.itemID.astype(str))
        n_imp = item_stats.impressions.fillna(0).to_numpy(dtype=float)
        n_clk = item_stats.clicks.fillna(0).to_numpy(dtype=float)
        ctr = (n_clk + self.prior*self.global_ctr)/(n_imp+self.prior)
        clicks = self.clicks.reindex(keys)
        counts = clicks['clicks'].fillna(0).to_numpy(dtype=float)
        last = clicks['last_click'].to_numpy(dtype=float)
        age_days = (timestamps.astype(float)-last)/SECONDS_PER_DAY
        recent = np.zeros(len(pairs),dtype=float)
        click_decay = np.zeros(len(pairs),dtype=float)
        lam = np.log(2.0)/(self.click_half_life_days*SECONDS_PER_DAY)
        # Unique user-item pairs are usually far fewer than full-catalog pairs.
        for j,(u,i,t) in enumerate(zip(pairs.user_id.astype(str),
                                      pairs.itemID.astype(str),timestamps)):
            arr = self.click_times.get((u,i))
            if arr is not None:
                valid = arr[arr < t]  # no current/future click used
                if len(valid):
                    click_decay[j] = np.exp(-lam*(t-valid)).sum()
                    recent[j] = 1.0/(1.0 + (t-valid[-1])/SECONDS_PER_DAY)
        # A repeated click signals persistent preference, with a separate
        # decayed component for interest drift. Zero history => popularity fallback.
        score = (self.ctr_weight*np.log(np.maximum(ctr,self.eps))
                 +self.click_weight*np.log1p(counts)
                 +self.recency_weight*np.log1p(click_decay)
                 +0.10*self.recency_weight*recent)
        if self.exposure_weight:
            seen = self.exposures.reindex(keys).fillna(0).to_numpy(dtype=float)
            # Experimental inverted-U proxy. Fit/ablate; not a causal fatigue law.
            exposure_curve = np.log1p(seen)-0.4*seen
            score += self.exposure_weight*exposure_curve
        return score.astype(float)

    def choose_available(self, user_id: str, timestamp: int) -> str:
        """Pick a valid catalogue item at a timestamp, with deterministic ties."""
        if not self._ready:
            raise RuntimeError('Call fit first')
        ts = int(timestamp)
        legal = self.catalog.loc[(self.catalog.valid_from <= ts)
                                 & (ts < self.catalog.valid_to)]
        if legal.empty:
            raise ValueError(f'No available catalog item at {ts}')
        items = legal.index.to_numpy(dtype=str)
        pairs = pd.DataFrame({'user_id':str(user_id),'timestamp':ts,'itemID':items})
        values = self.score(pairs)
        return str(items[np.argmax(values)])


def main():
    ap=argparse.ArgumentParser(description='Test Lea SequenceRanker on Huawei only')
    ap.add_argument('data_dir',help='Folder containing feedback_long.csv and catalog.csv')
    ap.add_argument('--days',type=float,default=1)
    ap.add_argument('--folds',type=int,default=2)
    args=ap.parse_args()
    from validation import load, rolling_splits, compute_metrics, SmoothedCTR
    feedback,catalog=load(args.data_dir)
    configs=[
        ('CTR baseline',None),
        ('Lea frequency',dict(click_weight=1.0,recency_weight=0.0)),
        ('Lea frequency+drift 1d',dict(click_half_life_days=1.0)),
        ('Lea frequency+drift 3d',dict(click_half_life_days=3.0)),
        ('Lea frequency+drift 7d',dict(click_half_life_days=7.0)),
        ('Lea drift+exposure',dict(click_half_life_days=3.0,exposure_weight=0.10)),
    ]
    outcome=[]
    for fold,train,test,cutoff in rolling_splits(feedback,args.days,args.folds):
        mask=test.groupby('event_id')['clicked'].transform('sum').eq(1)
        slate=test.loc[mask].reset_index(drop=True)
        if slate.empty: continue
        pairs=slate[['user_id','itemID','timestamp']]
        print(f'Fold {fold}: {slate.event_id.nunique()} clicked slates',flush=True)
        for name,parameters in configs:
            if parameters is None:
                m=SmoothedCTR(half_life_days=1.0,prior_strength=100.0)
            else:
                m=LeaSequenceRanker(**parameters)
            m.fit(train,catalog)
            scored=m.score(pairs)
            if not np.isfinite(scored).all():
                raise ValueError(f'Nonfinite score from {name}')
            result=compute_metrics(slate,scored,set(train.user_id.unique()))
            print(f'  {name:25s} hit@1={result["hit1"]:.4f} '
                  f'MRR={result["mrr"]:.4f}',flush=True)
            outcome.append(dict(fold=fold,name=name,n=result['n_events'],
                                hit1=result['hit1'],mrr=result['mrr']))
    if not outcome:
        raise RuntimeError('No evaluable folds')
    print('\nEvent-weighted results (shown-slate ranking, NOT official score):')
    for name,group in pd.DataFrame(outcome).groupby('name',sort=False):
        weights=group.n.to_numpy()
        print(f'  {name:25s} hit@1={np.average(group.hit1,weights=weights):.4f} '
              f'MRR={np.average(group.mrr,weights=weights):.4f}')
    print('CAUTION: select hyperparameters on one period; confirm on a separate future holdout.')

if __name__=='__main__':
    main()
