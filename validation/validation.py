
#!/usr/bin/env python3
"""Offline validation toolkit for Tech Arena Phase 2.

Models implement fit(train, catalog) and score(pairs).
Ranking metrics use historical events with exactly one click.
These metrics are not the official simulator score.
"""

import argparse
import itertools
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from diagnostics import print_diagnostics

DAY = 86_400
HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_CSV = os.path.join(HERE, "experiments.csv")

_CACHE = {}


def load(data_dir):
    """Load historical feedback and the Round 1 catalog."""
    data_dir = os.path.abspath(os.path.expanduser(data_dir))

    if data_dir not in _CACHE:
        fl = pd.read_csv(
            os.path.join(data_dir, "feedback_long.csv"),
            dtype={
                "event_id": str,
                "user_id": str,
                "itemID": str,
                "timestamp": "int64",
                "clicked": "int8",
            },
        )

        cat = pd.read_csv(
            os.path.join(data_dir, "catalog.csv"),
            dtype={
                "itemID": str,
                "valid_from": "int64",
                "valid_to": "int64",
            },
        )

        _CACHE[data_dir] = (fl, cat)

    return _CACHE[data_dir]


def time_split(fl, days=5):
    """Split feedback into historical training and recent test data."""
    if days <= 0:
        raise ValueError("days must be positive")

    cutoff = int(fl["timestamp"].max()) - int(days * DAY)

    train = fl[fl["timestamp"] < cutoff].reset_index(drop=True)
    test = fl[fl["timestamp"] >= cutoff].reset_index(drop=True)

    return train, test, cutoff


def rolling_splits(fl, days=1, folds=2):
    """Generate non-overlapping chronological validation windows."""
    if days <= 0 or folds < 1:
        raise ValueError("days must be positive and folds at least 1")

    window = int(days * DAY)
    end = int(fl["timestamp"].max()) + 1

    for fold in range(folds):
        cutoff = end - (folds - fold) * window

        train = fl[
            fl["timestamp"] < cutoff
        ].reset_index(drop=True)

        test = fl[
            (fl["timestamp"] >= cutoff)
            & (fl["timestamp"] < cutoff + window)
        ].reset_index(drop=True)

        if train.empty or test.empty:
            print(f"Skipping fold {fold + 1}: empty split")
            continue

        yield fold + 1, train, test, cutoff


def compute_metrics(test, scores, known_users):
    """Calculate pessimistic Hit@1 and MRR on single-click events."""
    df = test[["event_id", "user_id", "clicked"]].copy()
    df["s"] = np.asarray(scores, dtype=float)

    n_clicks = df.groupby("event_id")["clicked"].transform("sum")
    df = df[n_clicks == 1].copy()

    if df.empty:
        raise ValueError("No single-click events for ranking metrics")

    clicked_s = (
        df.loc[df["clicked"] == 1]
        .set_index("event_id")["s"]
    )

    df["cs"] = df["event_id"].map(clicked_s)

    df["beats"] = (
        (df["clicked"] == 0)
        & (df["s"] >= df["cs"])
    )

    ev = df.groupby("event_id").agg(
        user_id=("user_id", "first"),
        beats=("beats", "sum"),
    )

    ev["rank"] = 1 + ev["beats"]
    ev["hit"] = (ev["rank"] == 1).astype(float)
    ev["rr"] = 1.0 / ev["rank"]
    ev["known"] = ev["user_id"].isin(known_users)

    def part(mask):
        sub = ev[mask]
        return (
            sub["hit"].mean() if len(sub) else np.nan,
            sub["rr"].mean() if len(sub) else np.nan,
            len(sub),
        )

    h, r, n = part(slice(None))
    hk, rk, nk = part(ev["known"])
    hn, rn, nn = part(~ev["known"])

    return {
        "hit1": h,
        "mrr": r,
        "n_events": n,
        "hit1_known": hk,
        "mrr_known": rk,
        "n_known": nk,
        "hit1_new": hn,
        "mrr_new": rn,
        "n_new": nn,
    }


def evaluate_split(
    model, train, test, catalog,
    params=None, name=None, fold=None
):
    """Fit a model and evaluate one chronological split."""
    params = dict(params or {})

    if train.empty or test.empty:
        raise ValueError("Training and test data must not be empty")

    total_events = test["event_id"].nunique()

    n_clicks = test.groupby("event_id")["clicked"].transform("sum")
    ranking_test = test[n_clicks == 1].reset_index(drop=True)

    ranking_events = ranking_test["event_id"].nunique()

    if ranking_events == 0:
        raise ValueError("No single-click events in validation window")

    pairs = ranking_test[
        ["user_id", "timestamp", "itemID"]
    ].copy()

    m = model(**params) if isinstance(model, type) else model
    name = name or type(m).__name__

    t0 = time.perf_counter()
    m.fit(train.copy(), catalog.copy())
    t1 = time.perf_counter()

    scores = np.asarray(m.score(pairs), dtype=float).ravel()
    t2 = time.perf_counter()

    if len(scores) != len(pairs):
        raise ValueError(
            f"{name}.score returned {len(scores)} values "
            f"for {len(pairs)} rows"
        )

    if not np.isfinite(scores).all():
        raise ValueError(f"{name}.score returned non-finite values")

    res = compute_metrics(
        ranking_test,
        scores,
        set(train["user_id"].unique()),
    )

    res["total_test_events"] = total_events
    res["ranking_coverage"] = ranking_events / total_events
    res["no_click_events"] = int(
        (test.groupby("event_id")["clicked"].sum() == 0).sum()
    )
    res["fold"] = fold

    pstr = json.dumps(params, sort_keys=True) if params else ""

    print(
        f"{name:<12} {pstr:<45} "
        f"hit@1={res['hit1']:.4f} "
        f"MRR={res['mrr']:.4f} "
        f"coverage={res['ranking_coverage']:.1%} "
        f"known hit@1={res['hit1_known']:.4f} "
        f"(n={res['n_known']}) "
        f"new hit@1={res['hit1_new']:.4f} "
        f"(n={res['n_new']}) "
        f"[{t1 - t0:.1f}s fit, {t2 - t1:.1f}s score]"
    )

    return {
        "model": name,
        "params": pstr,
        **res,
        "fit_sec": round(t1 - t0, 2),
        "score_sec": round(t2 - t1, 2),
    }


def log_experiment(result, data_dir, days):
    """Append results while retaining existing experiment columns."""
    row = {
        "run_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "data_dir": os.path.abspath(os.path.expanduser(data_dir)),
        "days": days,
        **result,
    }

    new = pd.DataFrame([row])

    if os.path.exists(EXPERIMENTS_CSV):
        previous = pd.read_csv(EXPERIMENTS_CSV)
        new = pd.concat([previous, new], ignore_index=True)

    new.to_csv(EXPERIMENTS_CSV, index=False)


def evaluate(model, data_dir, days=5, params=None, name=None, log=True):
    """Evaluate a model on the most recent historical window."""
    fl, cat = load(data_dir)
    train, test, cutoff = time_split(fl, days)

    if train.empty or test.empty:
        raise ValueError(
            f"Empty split: train={len(train)}, test={len(test)}"
        )

    result = evaluate_split(
        model, train, test, cat,
        params=params, name=name,
    )
    result["cutoff"] = cutoff

    if log:
        log_experiment(result, data_dir, days)

    return result


def evaluate_rolling(
    model, data_dir, days=1, folds=2,
    params=None, name=None, log=True
):
    """Evaluate a model across expanding-history time splits."""
    fl, cat = load(data_dir)
    results = []

    for fold, train, test, cutoff in rolling_splits(
        fl, days=days, folds=folds
    ):
        print(f"\nFold {fold}")

        try:
            result = evaluate_split(
                model, train, test, cat,
                params=params, name=name, fold=fold,
            )
        except ValueError as error:
            print(f"Skipping fold {fold}: {error}")
            continue

        result["cutoff"] = cutoff
        results.append(result)

        if log:
            log_experiment(result, data_dir, days)

    if not results:
        raise ValueError("No usable rolling validation folds")

    df = pd.DataFrame(results)
    weights = df["n_events"].to_numpy(dtype=float)

    hit1 = np.average(df["hit1"], weights=weights)
    mrr = np.average(df["mrr"], weights=weights)

    print(
        f"\nRolling results: {len(df)} folds, "
        f"hit@1={hit1:.4f}, MRR={mrr:.4f}"
    )

    return df


def leaderboard(data_dir=None, top=None):
    """Print logged runs sorted by Hit@1."""
    if not os.path.exists(EXPERIMENTS_CSV):
        print("No experiments logged yet.")
        return None

    df = pd.read_csv(
        EXPERIMENTS_CSV,
        keep_default_na=False,
        na_values=[""],
    )

    if data_dir:
        path = os.path.abspath(os.path.expanduser(data_dir))
        df = df[df["data_dir"] == path]

    if df.empty:
        print("No matching experiments")
        return df

    df = df.sort_values(
        ["hit1", "mrr"], ascending=False
    ).reset_index(drop=True)

    if top:
        df = df.head(top)

    cols = [
        "model", "params", "days", "fold",
        "hit1", "mrr", "ranking_coverage",
        "hit1_known", "hit1_new",
        "n_events", "run_at",
    ]

    cols = [col for col in cols if col in df.columns]
    display = df[cols].copy()

    with pd.option_context(
        "display.width", 220,
        "display.max_rows", 100,
        "display.max_colwidth", 60,
    ):
        print(display.to_string(index=False))

    return df


class Random:
    """Random scoring baseline."""

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, train, catalog):
        pass

    def score(self, pairs):
        return np.random.default_rng(self.seed).random(len(pairs))


class ClickCount:
    """Score items using their historical click counts."""

    def fit(self, train, catalog):
        self.counts = train.groupby("itemID")["clicked"].sum()

    def score(self, pairs):
        return (
            pairs["itemID"]
            .map(self.counts)
            .fillna(0)
            .to_numpy(float)
        )


class SmoothedCTR:
    """Time-decayed item CTR with Bayesian smoothing."""

    def __init__(self, half_life_days=7.0, prior_strength=20.0):
        if half_life_days is not None and half_life_days <= 0:
            raise ValueError("half_life_days must be positive")

        if prior_strength <= 0:
            raise ValueError("prior_strength must be positive")

        self.half_life_days = half_life_days
        self.prior_strength = prior_strength

    def fit(self, train, catalog):
        age = (
            train["timestamp"].max() - train["timestamp"]
        ) / DAY

        if self.half_life_days is None:
            weights = np.ones(len(train))
        else:
            weights = np.asarray(
                0.5 ** (age / self.half_life_days),
                dtype=float,
            )

        d = pd.DataFrame({
            "itemID": train["itemID"].to_numpy(),
            "w": weights,
            "wc": weights * train["clicked"].to_numpy(),
        })

        g = d.groupby("itemID")[["w", "wc"]].sum()

        self.global_ctr = g["wc"].sum() / g["w"].sum()

        self.ctr = (
            g["wc"] + self.prior_strength * self.global_ctr
        ) / (
            g["w"] + self.prior_strength
        )

    def score(self, pairs):
        return (
            pairs["itemID"]
            .map(self.ctr)
            .fillna(self.global_ctr)
            .to_numpy(float)
        )


BASELINES = [
    (Random, {}),
    (ClickCount, {}),
    (
        SmoothedCTR,
        {"half_life_days": 7, "prior_strength": 20},
    ),
]

TUNE_GRID = {
    "half_life_days": [1, 3, 7, 14, None],
    "prior_strength": [1, 5, 20, 100],
}


def main():
    parser = argparse.ArgumentParser(
        description="Offline validation for Tech Arena Phase 2"
    )

    parser.add_argument("data_dir")
    parser.add_argument("--days", type=float, default=2)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--rolling", action="store_true")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument(
        "--skip-diagnostics", action="store_true"
    )

    args = parser.parse_args()

    fl, cat = load(args.data_dir)

    fmt = lambda t: datetime.fromtimestamp(
        int(t), timezone.utc
    ).strftime("%Y-%m-%d %H:%M")

    print(
        f"Data: {fl['event_id'].nunique():,} events, "
        f"{len(cat):,} catalog items, "
        f"{fmt(fl['timestamp'].min())} to "
        f"{fmt(fl['timestamp'].max())} UTC"
    )

    if not args.skip_diagnostics:
        print_diagnostics(fl, cat, args.data_dir)

    if not args.rolling:
        train, test, cutoff = time_split(fl, args.days)

        if train.empty or test.empty:
            raise ValueError("Empty chronological split")

        clicks = test.groupby("event_id")["clicked"].sum()
        known = set(train["user_id"])
        new_users = (
            ~test["user_id"].drop_duplicates().isin(known)
        ).sum()

        print(
            f"\nSplit at {fmt(cutoff)}: "
            f"train {train['event_id'].nunique():,} events, "
            f"test {test['event_id'].nunique():,} events, "
            f"{(clicks == 1).sum():,} single-click events, "
            f"{(clicks == 0).sum():,} no-click events, "
            f"{new_users:,} new users"
        )

    def run_model(cls, params):
        if args.rolling:
            return evaluate_rolling(
                cls, args.data_dir,
                days=args.days,
                folds=args.folds,
                params=params,
            )

        return evaluate(
            cls, args.data_dir,
            days=args.days,
            params=params,
        )

    print("\nBaseline results")

    for cls, params in BASELINES:
        run_model(cls, params)

    if args.tune:
        print("\nSmoothedCTR tuning")

        keys = list(TUNE_GRID)
        existing = [
            params for cls, params in BASELINES
            if cls is SmoothedCTR
        ]

        for combo in itertools.product(*TUNE_GRID.values()):
            params = dict(zip(keys, combo))

            if params not in existing:
                run_model(SmoothedCTR, params)

        print(
            "\nTuned parameters require confirmation "
            "on a separate chronological holdout."
        )

    print("\nExperiment leaderboard")
    leaderboard(args.data_dir)

    print(
        "\nHistorical ranking metrics are not "
        "the official Huawei simulator score."
    )


if __name__ == "__main__":
    main()
