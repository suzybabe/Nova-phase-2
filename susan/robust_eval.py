"""Robustness check: 6 chronological splits of the round-2 feedback (different history ends + gaps).
Compares Lea's frequency rule, v4 and v5 on real slates (slate hit@1, ties count against).
Run ON THE VM:  python3 robust_eval.py ~/data/round-2/phase-2-round-2-input
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import solution as S

DAY = 86400
fb = S.load_feedback(Path(sys.argv[1]).expanduser())
END = fb["timestamp"].max() + 1


def hit(df, s):
    d = pd.DataFrame({"e": pd.factorize(df["rowid"])[0], "s": s, "y": df["clicked"].to_numpy()})
    d["r"] = d.groupby("e")["s"].rank(ascending=False, method="max")
    p = d[d["y"] == 1]
    return (p["r"] == 1).mean(), (1 / p["r"]).mean()


def model_scores(h, rows, sl, pooled_only, clock_at_end):
    S.POOLED_ONLY, S.CLOCK_AT_END = pooled_only, clock_at_end
    items = pd.unique(h["itemID"])
    appeal = pd.Series(S.global_scores(h, items), index=items)
    trans = S.transitions(h)
    tr, loo = S.train_set(h, trans)
    w, mu, sd = S.fit_logit(S.features(tr, appeal, trans, loo), pd.factorize(tr["event_id"])[0],
                            tr["clicked"].to_numpy(float))
    st, u, tend = S.end_state(h)
    df = S.pred_frame(rows, sl[["rowid", "itemID"]], st, u, tend).merge(sl, on=["rowid", "itemID"])
    return df, ((S.features(df, appeal, trans) - mu) / sd) @ w, appeal, tend


tot = {}
print(f"{'history <':<12}{'test day':<12}{'slates':>7}{'freq rule':>11}{'v4':>8}{'v5':>8}")
for hist_days, test_day in [(2, 0), (2, 1), (1, 0), (3, 0), (3, 2), (4, 3)]:
    cut = END - hist_days * DAY
    t0 = END - (test_day + 1) * DAY
    h = fb[fb["timestamp"] < cut]
    w = fb[(fb["timestamp"] >= t0) & (fb["timestamp"] < t0 + DAY)]
    ev = w.groupby("event_id")["clicked"].sum()
    w = w[w["event_id"].isin(ev[ev == 1].index)]
    rows = w.drop_duplicates("event_id")[["event_id", "user_id", "timestamp"]].rename(columns={"event_id": "rowid"})
    sl = w[["event_id", "itemID", "clicked"]].rename(columns={"event_id": "rowid"})
    df4, s4, appeal, tend = model_scores(h, rows, sl, False, False)
    df5, s5, _, _ = model_scores(h, rows, sl, True, True)
    cl = h[h["clicked"] == 1]
    g = pd.Series(np.exp(-(tend - cl["timestamp"].to_numpy(float)) / (192 * 3600)), index=cl.index)
    wf = df4.join(g.groupby([cl["user_id"], cl["itemID"]]).sum().rename("wf"), on=["user_id", "itemID"])["wf"]
    nc, lc = df4["n_clk"].fillna(0).to_numpy(), df4["last_clk"].fillna(0).to_numpy()
    rule = np.where(nc > 0, 1e6 + wf.fillna(0).to_numpy() * 1e3 + lc / 1e7,
                    df4["itemID"].map(appeal).fillna(appeal.min()).to_numpy())
    r = {"freq rule": hit(df4, rule), "v4": hit(df4, s4), "v5": hit(df5, s5)}
    n = rows.shape[0]
    for k, v in r.items():
        a = tot.setdefault(k, [0.0, 0.0, 0])
        a[0] += v[0] * n; a[1] += v[1] * n; a[2] += n
    print(f"{str(pd.to_datetime(cut, unit='s').date()):<12}{str(pd.to_datetime(t0, unit='s').date()):<12}{n:>7}"
          f"{r['freq rule'][0]:>11.4f}{r['v4'][0]:>8.4f}{r['v5'][0]:>8.4f}")
print("\nweighted over all 6 splits:")
for k, (h1, m, n) in tot.items():
    print(f"  {k:<10} hit@1 {h1 / n:.4f}   MRR {m / n:.4f}")