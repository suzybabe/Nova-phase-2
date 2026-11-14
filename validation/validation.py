#!/usr/bin/env python3
"""Offline validation toolkit for Tech Arena Phase 2 (one recommended item per request).

Usage:
    python3 validation.py DATA_DIR [--days N] [--tune]

From Python:
    from validation import evaluate, leaderboard, SmoothedCTR
    evaluate(SmoothedCTR, DATA_DIR, days=5, params={"half_life_days": 3, "prior_strength": 20})
    leaderboard()

MODEL CONTRACT (what teammates implement):
    class MyModel:
        def __init__(self, **params): ...
        def fit(self, train, catalog): ...
            # train:   DataFrame event_id, user_id, timestamp, itemID, clicked  (feedback_long rows)
            # catalog: DataFrame itemID, valid_from, valid_to
        def score(self, pairs) -> array-like of len(pairs):
            # pairs:   DataFrame user_id, timestamp, itemID  (no click info!)
            # return one finite number per row, higher = more likely to be clicked

Metrics are computed on test events that had a click. Within each event the model
ranks the displayed items; ties count AGAINST the model (pessimistic rank).
"""
import argparse
import itertools
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

DAY = 86_400
HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_CSV = os.path.join(HERE, "experiments.csv")

_CACHE = {}


# ----------------------------------------------------------------------------- data
def load(data_dir):
    """Load feedback_long.csv and catalog.csv (IDs as strings, timestamps as int64). Cached."""
    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    if data_dir not in _CACHE:
        fl = pd.read_csv(os.path.join(data_dir, "feedback_long.csv"),
                         dtype={"event_id": str, "user_id": str, "itemID": str,
                                "timestamp": "int64", "clicked": "int8"})
        cat = pd.read_csv(os.path.join(data_dir, "catalog.csv"),
                          dtype={"itemID": str, "valid_from": "int64", "valid_to": "int64"})
        _CACHE[data_dir] = (fl, cat)
    return _CACHE[data_dir]


def time_split(fl, days=5):
    """Train = everything before the last `days` days, test = the last `days` days."""
    cutoff = int(fl["timestamp"].max()) - int(days * DAY)
    train = fl[fl["timestamp"] < cutoff].reset_index(drop=True)
    test = fl[fl["timestamp"] >= cutoff].reset_index(drop=True)
    return train, test, cutoff


# ----------------------------------------------------------------------------- metrics
def compute_metrics(test, scores, known_users):
    """hit@1 and MRR over test events with exactly one click; ties count against the model."""
    df = test[["event_id", "user_id", "clicked"]].copy()
    df["s"] = scores
    n_clicks = df.groupby("event_id")["clicked"].transform("sum")
    df = df[n_clicks == 1]
    clicked_s = df.loc[df["clicked"] == 1].set_index("event_id")["s"]
    df["cs"] = df["event_id"].map(clicked_s)
    # pessimistic rank: 1 + number of OTHER displayed items scoring >= the clicked one
    df["beats"] = (df["clicked"] == 0) & (df["s"] >= df["cs"])
    ev = df.groupby("event_id").agg(user_id=("user_id", "first"), beats=("beats", "sum"))
    ev["rank"] = 1 + ev["beats"]
    ev["hit"] = (ev["rank"] == 1).astype(float)
    ev["rr"] = 1.0 / ev["rank"]
    ev["known"] = ev["user_id"].isin(known_users)

    def part(mask):
        sub = ev[mask]
        return (sub["hit"].mean() if len(sub) else np.nan,
                sub["rr"].mean() if len(sub) else np.nan, len(sub))

    h, r, n = part(slice(None))
    hk, rk, nk = part(ev["known"])
    hn, rn, nn = part(~ev["known"])
    return {"hit1": h, "mrr": r, "n_events": n,
            "hit1_known": hk, "mrr_known": rk, "n_known": nk,
            "hit1_new": hn, "mrr_new": rn, "n_new": nn}


# ----------------------------------------------------------------------------- evaluate
def evaluate(model, data_dir, days=5, params=None, name=None, log=True):
    """Train on all but the last `days` days, score the test window, print one line, log to experiments.csv.

    `model` may be a class (instantiated with **params) or an already-built instance.
    """
    params = dict(params or {})
    fl, cat = load(data_dir)
    train, test, cutoff = time_split(fl, days)
    if train.empty or test.empty:
        raise ValueError(f"empty split (train={len(train)}, test={len(test)}); check --days")

    # only score events that had exactly one click (the only ones the metrics use)
    n_clicks = test.groupby("event_id")["clicked"].transform("sum")
    test = test[n_clicks == 1].reset_index(drop=True)
    pairs = test[["user_id", "timestamp", "itemID"]].copy()   # no click info for the model

    m = model(**params) if isinstance(model, type) else model
    name = name or type(m).__name__

    t0 = time.time()
    m.fit(train.copy(), cat.copy())
    t1 = time.time()
    scores = np.asarray(m.score(pairs), dtype=float).ravel()
    t2 = time.time()
    if len(scores) != len(pairs):
        raise ValueError(f"{name}.score returned {len(scores)} values for {len(pairs)} rows")
    if not np.isfinite(scores).all():
        raise ValueError(f"{name}.score returned NaN/inf for {(~np.isfinite(scores)).sum()} rows")

    res = compute_metrics(test, scores, set(train["user_id"].unique()))
    pstr = json.dumps(params, sort_keys=True) if params else ""
    print(f"{name:<12} {pstr:<45} hit@1={res['hit1']:.4f}  MRR={res['mrr']:.4f}  "
          f"known hit@1={res['hit1_known']:.4f} (n={res['n_known']})  "
          f"new hit@1={res['hit1_new']:.4f} (n={res['n_new']})  [{t1 - t0:.1f}s fit, {t2 - t1:.1f}s score]")

    row = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
           "model": name, "params": pstr,
           "data_dir": os.path.abspath(os.path.expanduser(data_dir)), "days": days,
           **res, "fit_sec": round(t1 - t0, 2), "score_sec": round(t2 - t1, 2)}
    if log:
        pd.DataFrame([row]).to_csv(EXPERIMENTS_CSV, mode="a", index=False,
                                   header=not os.path.exists(EXPERIMENTS_CSV))
    return row


def leaderboard(data_dir=None, top=None):
    """Print all logged runs sorted by hit@1 (optionally only runs on one data_dir)."""
    if not os.path.exists(EXPERIMENTS_CSV):
        print("No experiments logged yet.")
        return None
    df = pd.read_csv(EXPERIMENTS_CSV, keep_default_na=False, na_values=[""])
    if data_dir:
        df = df[df["data_dir"] == os.path.abspath(os.path.expanduser(data_dir))]
    df = df.sort_values(["hit1", "mrr"], ascending=False).reset_index(drop=True)
    if top:
        df = df.head(top)
    cols = ["model", "params", "days", "hit1", "mrr", "hit1_known", "hit1_new", "n_events", "data_dir", "run_at"]
    df = df[cols].copy()
    df["data_dir"] = df["data_dir"].map(lambda p: os.path.join("…", *str(p).split(os.sep)[-3:]))
    df["params"] = df["params"].fillna("")
    with pd.option_context("display.width", 250, "display.max_colwidth", 60,
                           "display.max_rows", 500, "display.float_format", "{:.4f}".format):
        print(df.to_string())
    return df


# ----------------------------------------------------------------------------- baseline models
class Random:
    """Random scores. Anything useful must beat this clearly (expect hit@1 ≈ 1/10)."""

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, train, catalog):
        pass

    def score(self, pairs):
        return np.random.default_rng(self.seed).random(len(pairs))


class ClickCount:
    """Score = total clicks the item got in train (unseen items = 0)."""

    def fit(self, train, catalog):
        self.counts = train.groupby("itemID")["clicked"].sum()

    def score(self, pairs):
        return pairs["itemID"].map(self.counts).fillna(0).to_numpy(float)


class SmoothedCTR:
    """Recency-weighted click-through rate per item, shrunk toward the global CTR.

    score = (w_clicks + prior_strength * global_ctr) / (w_impressions + prior_strength)
    where each train row has weight 0.5 ** (age_days / half_life_days), age measured from
    the end of train. half_life_days=None means no decay.
    """

    def __init__(self, half_life_days=7.0, prior_strength=20.0):
        self.half_life_days = half_life_days
        self.prior_strength = prior_strength

    def fit(self, train, catalog):
        age = (train["timestamp"].max() - train["timestamp"]) / DAY
        w = np.ones(len(train)) if self.half_life_days is None else 0.5 ** (age / self.half_life_days)
        d = pd.DataFrame({"itemID": train["itemID"], "w": w, "wc": w * train["clicked"]})
        g = d.groupby("itemID")[["w", "wc"]].sum()
        self.global_ctr = g["wc"].sum() / g["w"].sum()
        self.ctr = (g["wc"] + self.prior_strength * self.global_ctr) / (g["w"] + self.prior_strength)

    def score(self, pairs):
        return pairs["itemID"].map(self.ctr).fillna(self.global_ctr).to_numpy(float)


BASELINES = [(Random, {}), (ClickCount, {}), (SmoothedCTR, {"half_life_days": 7, "prior_strength": 20})]
TUNE_GRID = {"half_life_days": [1, 3, 7, 14, None], "prior_strength": [1, 5, 20, 100]}


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Offline validation for Tech Arena Phase 2")
    ap.add_argument("data_dir", help="folder with feedback_long.csv and catalog.csv")
    ap.add_argument("--days", type=float, default=5, help="size of the test window in days (default 5)")
    ap.add_argument("--tune", action="store_true", help="grid-search SmoothedCTR half_life/prior")
    a = ap.parse_args()

    fl, cat = load(a.data_dir)
    train, test, cutoff = time_split(fl, a.days)
    fmt = lambda t: datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%d %H:%M")
    clicked_ev = test.groupby("event_id")["clicked"].sum()
    new_users = ~test["user_id"].drop_duplicates().isin(set(train["user_id"]))
    print(f"Data: {fl['event_id'].nunique():,} events, {len(cat):,} catalog items, "
          f"{fmt(fl['timestamp'].min())} → {fmt(fl['timestamp'].max())} UTC")
    print(f"Split at {fmt(cutoff)}: train {train['event_id'].nunique():,} events | "
          f"test {test['event_id'].nunique():,} events, {(clicked_ev == 1).sum():,} with one click, "
          f"{(clicked_ev > 1).sum():,} with >1 click (skipped), {new_users.sum():,} new users in test")
    print()

    for cls, params in BASELINES:
        evaluate(cls, a.data_dir, a.days, params)
    if a.tune:
        print("\nTuning SmoothedCTR ...")
        keys = list(TUNE_GRID)
        already_run = [p for c, p in BASELINES if c is SmoothedCTR]
        for combo in itertools.product(*TUNE_GRID.values()):
            params = dict(zip(keys, combo))
            if params not in already_run:
                evaluate(SmoothedCTR, a.data_dir, a.days, params)
    print("\nLeaderboard (this data folder):")
    leaderboard(a.data_dir)


if __name__ == "__main__":
    main()
