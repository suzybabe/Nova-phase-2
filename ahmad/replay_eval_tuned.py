"""Pooled-slate replay validation (matches how Huawei scores).

Uses the CURRENT round's input folder only (its feedback already contains the previous
round's real pooled slates). Time split inside the feedback:
  history = events before CUTOFF (= last REPLAY_DAYS days of the previous round)
  test    = pooled events after CUTOFF where the user clicked
For each test event the policy picks 1 item from history only:
  matched = our item was in that pooled slate   (outcome known)
  share   = hits / matched  ~ P(our item chosen | click)   <- what the score measures
  hits/all = lower bound of the round score proxy
Availability is not re-checked here (the test slates are past events); check_solution.py does that.
Second test (fairest for a choice model): SLATE hit@1 = rank the items of each real test slate,
how often is the top one the item the user actually clicked.
Usage (on the VM):  python3 replay_eval.py ~/data/round-2/phase-2-round-2-input [days] [policy-filter]
"""
import sys
import time
from pathlib import Path

import pandas as pd

import solution_tuned as S

POLICIES = {
    "v4 seq model, all candidates":         dict(RECLICK_HOURS=None, PERSONAL_RANK="seq", SEQ_CANDIDATES="all", SKIP_REJECTED=True),
    "v4 seq model, clicked candidates":     dict(RECLICK_HOURS=None, PERSONAL_RANK="seq", SEQ_CANDIDATES="clicked", SKIP_REJECTED=True),
    "global only (no personal)":            dict(RECLICK_HOURS=0.0, PERSONAL_RANK="recent", SKIP_REJECTED=False),
    "personal 12h, recent (= round 1)":     dict(RECLICK_HOURS=12.0, PERSONAL_RANK="recent", SKIP_REJECTED=False),
    "personal all, recent":                 dict(RECLICK_HOURS=None, PERSONAL_RANK="recent", SKIP_REJECTED=False),
    "personal all, appeal":                 dict(RECLICK_HOURS=None, PERSONAL_RANK="appeal", SKIP_REJECTED=False),
    "personal all, count then recent":      dict(RECLICK_HOURS=None, PERSONAL_RANK="count_recent", SKIP_REJECTED=False),
    "personal all, learned":                dict(RECLICK_HOURS=None, PERSONAL_RANK="learned", SKIP_REJECTED=False),
    "personal all, learned + skip rejected": dict(RECLICK_HOURS=None, PERSONAL_RANK="learned", SKIP_REJECTED=True),
}


def main(data_dir, days=2.0, only=None):
    data = Path(data_dir).expanduser()
    fb = S.load_feedback(data)
    cutoff = int(fb["timestamp"].max() - days * 86400)
    hist = fb[fb["timestamp"] < cutoff]
    test = fb[fb["timestamp"] >= cutoff]
    ev = test.groupby("event_id")["clicked"].sum()
    test = test[test["event_id"].isin(ev[ev > 0].index)]
    slate = test.groupby("event_id")["itemID"].apply(set)
    chosen = test[test["clicked"] == 1].drop_duplicates("event_id").set_index("event_id")["itemID"]
    rows = test.drop_duplicates("event_id")[["event_id", "user_id", "timestamp"]].rename(columns={"event_id": "rowid"})
    items = pd.unique(pd.concat([fb["itemID"], S.read_any(data, "catalog", dtype={"itemID": str})["itemID"].astype(str)]))
    cat = pd.DataFrame({"itemID": items, "valid_from": 0, "valid_to": 2**62})   # availability checked elsewhere
    print(f"history events {hist['event_id'].nunique():,} | test events with a click {len(rows):,} | "
          f"avg slate size {test.groupby('event_id').size().mean():.2f}\n")

    allp = {}
    base = {k: getattr(S, k) for k in ["RECLICK_HOURS", "PERSONAL_RANK", "SKIP_REJECTED", "SEQ_CANDIDATES"]}
    for name, cfg in POLICIES.items():
        if only and only not in name:
            continue
        for k, v in {**base, **cfg}.items():
            setattr(S, k, v)
        t = time.time()
        picks = dict(p for part in S.recommend(hist, rows, cat) for p in part)
        allp[name] = picks
        hit = sum(it == chosen[r] for r, it in picks.items())
        mat = sum(it in slate[r] for r, it in picks.items())
        print(f"{name:<40} share={hit / max(mat, 1):.4f}  hits/all={hit / len(rows):.4f}  "
              f"matched={mat / len(rows):.1%}  [{time.time() - t:.1f}s]\n")

    # HEAD-TO-HEAD: events where both picks were in the same real slate but differ
    a, b = "v4 seq model, all candidates", "personal all, count then recent"
    if a in allp and b in allp:
        n = wa = wb = 0
        for r in rows["rowid"]:
            pa, pb = allp[a][r], allp[b][r]
            if pa != pb and pa in slate[r] and pb in slate[r]:
                n += 1; wa += chosen[r] == pa; wb += chosen[r] == pb
        print(f"HEAD-TO-HEAD on {n} shared slates: v4 seq chosen {wa / max(n, 1):.1%} | "
              f"count-then-recent chosen {wb / max(n, 1):.1%}")


def slate_test(data_dir, days=2.0):
    import numpy as np
    data = Path(data_dir).expanduser()
    fb = S.load_feedback(data)
    cut = int(fb["timestamp"].max() - days * 86400)
    hist, test = fb[fb["timestamp"] < cut], fb[fb["timestamp"] >= cut]
    ev = test.groupby("event_id")["clicked"].sum()
    test = test[test["event_id"].isin(ev[ev == 1].index)]
    rows = test.drop_duplicates("event_id")[["event_id", "user_id", "timestamp"]].rename(columns={"event_id": "rowid"})
    sl = test[["event_id", "itemID", "clicked"]].rename(columns={"event_id": "rowid"})
    items = pd.unique(hist["itemID"])
    appeal = pd.Series(S.global_scores(hist, items), index=items)
    trans = S.transitions(hist)
    tr, loo = S.train_rows(hist, trans)
    e, _ = pd.factorize(tr["event_id"])
    w, mu, sd = S.fit_logit(S.features(tr, appeal, trans, loo), e, tr["clicked"].to_numpy(float))
    st, u, tend = S.end_state(hist)
    df = S.pred_frame(rows, sl[["rowid", "itemID"]], st, u, tend).merge(sl, on=["rowid", "itemID"])
    a = df["itemID"].map(appeal).fillna(appeal.min()).to_numpy()
    nc, lc = df["n_clk"].fillna(0).to_numpy(), df["last_clk"].fillna(0).to_numpy()
    scorers = {"global appeal only": a,
               "most recent click (rule)": np.where(nc > 0, 1e6 + lc / 1e4, a),
               "most clicks, then recent (rule)": np.where(nc > 0, 1e6 + nc * 1e3 + lc / 1e7, a),
               "v4 seq model": ((S.features(df, appeal, trans) - mu) / sd) @ w}
    print(f"SLATE test: {rows.shape[0]:,} real slates from the last {days:g} days")
    for k, v in scorers.items():
        top = df.assign(s=v).sort_values("s", kind="stable").drop_duplicates("rowid", keep="last")
        print(f"slate hit@1  {k:<34}{top['clicked'].mean():.4f}")
    print("\nwhat the model learned (standardised weights):")
    for f, v in zip(S.FEATS, w):
        print(f"  {f:<22}{v:+.3f}")
    print()


if __name__ == "__main__":
    slate_test(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 2.0)
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 2.0, sys.argv[3] if len(sys.argv) > 3 else None)