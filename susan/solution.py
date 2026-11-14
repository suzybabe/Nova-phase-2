"""Team Nova - Phase 2 submission (v5: sequence-aware choice model, serving-matched).

Score = P(our item is chosen | the user clicks on the pooled slate of all teams' items).
v4 treats each user's interactions as a SEQUENCE and learns, from real pooled slates,
which item a user picks:
  - interest drift: time-decayed clicks (6 h / 2 day), hours since last click,
    clicks made since (sequence distance), is it the user's last click
  - repeated impressions: exposure buckets for never-clicked items (1,2,3-4,5-9,10+),
    impressions since the last click, hours since last shown (fatigue / recovery)
  - sequence transitions: last clicked item -> next item
  - global appeal (Darling's MNL + smoothed CTR)
v5 (robustness, checked on 6 chronological splits): train only on pooled-slate days
(days whose average slate has > 3.5 items, i.e. real multi-team slates) and compute features
at the end of the history ("clock at end") so serving features look like training features
even when requests are days after the history.
Candidates per request: every item the user interacted with + top global + transition items,
all LIVE at the request time. Conditional logit picks the best. Falls back to v3 rules on error.
Only numpy/pandas/scipy.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HALF_LIFE_DAYS = 1.0     # recency decay for the global scores
PRIOR_STRENGTH = 100.0   # smoothing of the click rate
LAM = 5.0                # L2 on MNL item scores
LAM_U = 0.3              # L2 on per-user no-click scores
TIME_BINS = 24           # no-click score bins over the feedback period
W_MNL = 0.7              # global score = W_MNL * mnl + (1 - W_MNL) * log(ctr)
TOP_K = 512              # global candidates checked per row before a full scan
DAY = 86400

RECLICK_HOURS = None     # None = no time limit on previously clicked items
PERSONAL_RANK = "seq"     # "seq" (v4) | "learned" | "recent" | "appeal" | "count_recent"
SEQ_CANDIDATES = "all"   # "all" = interacted + global + transitions ; "clicked" = previously clicked only
N_GLOBAL = 50            # global candidates per request (seq model)
N_TRANS = 10             # transition candidates per request (seq model)
L2 = 1.0                 # L2 on the choice-model weights
POOLED_ONLY = True       # v5: train the choice model on pooled-slate days only
POOLED_MIN_SLATE = 3.5   # a day counts as pooled if its average slate has more items than this
CLOCK_AT_END = True      # v5: compute time features at the end of the history, not at the request
MIN_CHOOSER_EVENTS = 300  # below this the learned chooser falls back to "recent"
SKIP_REJECTED = True     # global fallback skips items the user saw and never clicked


# ------------------------------------------------------------------ loading
def read_any(data, name, **kw):
    pq, csv = data / (name + ".parquet"), data / (name + ".csv")
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception as e:
            print("parquet read failed for", name, e)
    return pd.read_csv(csv, **kw)


def load_feedback(data):
    try:
        fb = read_any(data, "feedback_long", dtype={"itemID": str, "user_id": str, "event_id": str})
    except Exception as e:
        print("no feedback_long, reading jsonl:", e)
        rows = []
        with open(data / "feedback.jsonl") as f:
            for line in f:
                e = json.loads(line)
                for it in e["displayed_items"]:
                    rows.append((e["event_id"], e["user_id"], e["timestamp"], it, int(it == e.get("clicked_item"))))
        fb = pd.DataFrame(rows, columns=["event_id", "user_id", "timestamp", "itemID", "clicked"])
    fb = fb.astype({"event_id": str, "user_id": str, "itemID": str})
    fb["timestamp"] = fb["timestamp"].astype("int64")
    fb["clicked"] = fb["clicked"].astype(int)
    return fb.drop_duplicates(["event_id", "itemID"])


# ------------------------------------------------------------------ global scores
def smoothed_ctr(fb, items):
    ts = fb["timestamp"].to_numpy()
    w = 0.5 ** ((ts.max() - ts) / (HALF_LIFE_DAYS * DAY))
    g = pd.DataFrame({"itemID": fb["itemID"].to_numpy(), "w": w,
                      "wc": w * fb["clicked"].to_numpy(float)}).groupby("itemID")[["w", "wc"]].sum()
    g_ctr = fb["clicked"].mean()
    ctr = (g["wc"] + PRIOR_STRENGTH * g_ctr) / (g["w"] + PRIOR_STRENGTH)
    return ctr.reindex(items).fillna(g_ctr).to_numpy()


def fit_mnl(fb):
    from scipy.optimize import minimize
    fb = fb.sort_values("event_id").reset_index(drop=True)
    ev, _ = pd.factorize(fb["event_id"])
    it, item_ids = pd.factorize(fb["itemID"])
    E, I = ev.max() + 1, len(item_ids)
    clk = fb["clicked"].to_numpy(float)
    first = fb.groupby(ev).first()
    et = first["timestamp"].to_numpy(dtype=float)
    eu, users = pd.factorize(first["user_id"])
    U = len(users)
    eclk = np.bincount(ev, weights=clk, minlength=E)
    we = 0.5 ** ((et.max() - et) / (HALF_LIFE_DAYS * DAY))
    tb = np.minimum(((et - et.min()) / (et.max() - et.min() + 1) * TIME_BINS).astype(int), TIME_BINS - 1)

    def loss(x):
        b, c, cu = x[:I], x[I:I + TIME_BINS], x[I + TIME_BINS:]
        eb = np.exp(b[it])
        ce = c[tb] + cu[eu]
        eo = np.exp(ce)
        den = np.bincount(ev, weights=eb, minlength=E) + eo
        ll = np.bincount(ev, weights=clk * b[it], minlength=E) + (1 - eclk) * ce - np.log(den)
        f = -(we * ll).sum() + 0.5 * LAM * (b @ b) + 0.5 * LAM_U * (cu @ cu)
        gb = -np.bincount(it, weights=we[ev] * (clk - eb / den[ev]), minlength=I) + LAM * b
        r = we * ((1 - eclk) - eo / den)
        gc = -np.bincount(tb, weights=r, minlength=TIME_BINS)
        gu = -np.bincount(eu, weights=r, minlength=U) + LAM_U * cu
        return f, np.concatenate([gb, gc, gu])

    x0 = np.zeros(I + TIME_BINS + U)
    x0[I:I + TIME_BINS] = 2.0
    res = minimize(loss, x0, jac=True, method="L-BFGS-B", options={"maxiter": 300})
    return pd.Series(res.x[:I], index=item_ids)


def global_scores(fb, items):
    ctr = smoothed_ctr(fb, items)
    try:
        b = fit_mnl(fb).reindex(items).fillna(0.0).to_numpy()
        s = W_MNL * b + (1 - W_MNL) * np.log(ctr)
        if np.isfinite(s).all():
            return s
        print("non-finite MNL scores, using CTR")
    except Exception as e:
        print("MNL failed, using CTR:", e)
    return np.log(ctr)



# ------------------------------------------------------------------ learned chooser
def _cand_features(df, appeal):
    """df: rows of (key, itemID, ts, last, n) -> feature matrix for previously clicked candidates."""
    hrs = (df["ts"].to_numpy(float) - df["last"].to_numpy(float)) / 3600.0
    rank = df.groupby("key")["last"].rank(ascending=False, method="first").to_numpy()
    return np.column_stack([
        np.log1p(np.maximum(hrs, 0.0)),
        np.log1p(df["n"].to_numpy(float)),
        df["itemID"].map(appeal).fillna(appeal.min() if len(appeal) else 0.0).to_numpy(float),
        (rank == 1).astype(float),
    ])


def fit_chooser(fb, appeal):
    """Conditional logit over the user's previously clicked items, learned from feedback events
    where the user clicked one of >=2 previously clicked items shown. Returns weights or None."""
    from scipy.optimize import minimize
    ev_click = fb.groupby("event_id")["clicked"].transform("sum")
    imp = fb[ev_click > 0][["event_id", "user_id", "itemID", "timestamp", "clicked"]].sort_values("timestamp")
    clk = fb[fb["clicked"] == 1][["user_id", "itemID", "timestamp"]].sort_values("timestamp").copy()
    clk["n"] = clk.groupby(["user_id", "itemID"]).cumcount() + 1
    clk = clk.rename(columns={"timestamp": "last"})
    clk["timestamp"] = clk["last"]
    m = pd.merge_asof(imp, clk, on="timestamp", by=["user_id", "itemID"], allow_exact_matches=False)
    m = m[m["last"].notna()]
    size = m.groupby("event_id")["itemID"].transform("size")
    pos = m.groupby("event_id")["clicked"].transform("sum")
    m = m[(size >= 2) & (pos == 1)].rename(columns={"event_id": "key", "timestamp": "ts"})
    n_ev = m["key"].nunique()
    print(f"chooser training events: {n_ev}")
    if n_ev < MIN_CHOOSER_EVENTS:
        return None
    X = _cand_features(m, appeal)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    X = (X - mu) / sd
    ev, _ = pd.factorize(m["key"])
    y = m["clicked"].to_numpy(float)
    E = ev.max() + 1

    def loss(w):
        u = X @ w
        mx = np.full(E, -np.inf); np.maximum.at(mx, ev, u)
        e = np.exp(u - mx[ev]); s = np.bincount(ev, weights=e, minlength=E)
        p = e / s[ev]
        f = -(y * (u - mx[ev] - np.log(s[ev]))).sum() + 0.5 * 1e-2 * (w @ w)
        g = X.T @ (p - y) + 1e-2 * w
        return f, g

    res = minimize(loss, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
    print("chooser weights (recency_h, count, appeal, is_most_recent):", np.round(res.x, 3))
    return res.x, mu, sd

# ------------------------------------------------------------------ personal picks
def personal_picks(fb, rows, live, appeal, chooser=None, rank=None):
    """rowid -> previously clicked item of that user that is live at the row's time."""
    clk = fb[fb["clicked"] == 1][["user_id", "itemID", "timestamp"]]
    hist = (clk.groupby(["user_id", "itemID"])["timestamp"].agg(last="max", n="size").reset_index())
    m = rows[["rowid", "user_id", "timestamp"]].merge(hist, on="user_id")
    if m.empty:
        return {}
    m = m[m["last"] <= m["timestamp"]]                       # only clicks before the request
    if RECLICK_HOURS is not None:
        m = m[m["timestamp"] - m["last"] <= RECLICK_HOURS * 3600]
    vf = m["itemID"].map(live["valid_from"])
    vt = m["itemID"].map(live["valid_to"])
    m = m[vf.notna() & (vf <= m["timestamp"]) & (m["timestamp"] < vt)]
    if m.empty:
        return {}
    rank = rank or PERSONAL_RANK
    if rank == "learned" and chooser is not None:
        w, mu, sd = chooser
        f = m.rename(columns={"rowid": "key", "timestamp": "ts"})
        m = m.assign(key=((_cand_features(f, appeal) - mu) / sd) @ w).sort_values(["key", "last"])
    elif rank == "appeal":
        m = m.assign(key=m["itemID"].map(appeal).fillna(-1e9)).sort_values(["key", "last"])
    elif rank == "count_recent":
        m = m.sort_values(["n", "last"])
    else:
        m = m.sort_values("last")
    m = m.drop_duplicates("rowid", keep="last")
    return dict(zip(m["rowid"], m["itemID"]))


def rejected_pairs(fb):
    """(user, item) pairs the user was shown but never clicked."""
    g = fb.groupby(["user_id", "itemID"])["clicked"].max()
    g = g[g == 0]
    return set(g.index)



# ================================================================== v4 sequence model
H = 3600.0
FEATS = ["appeal", "clicked_before", "log_clicks", "dec6h", "dec2d", "log_h_since_click",
         "log_clicks_since", "is_last_click", "imp_since_click_0", "imp_since_click_1",
         "imp_since_click_2", "imp_since_click_3_5", "imp_since_click_6p",
         "seen_1", "seen_2", "seen_3_4", "seen_5_9", "seen_10p", "log_h_since_shown", "transition"]
HALF = {"dec6h": 6 * H, "dec2d": 48 * H}


# --------------------------------------------------------------- per-row history state
def _ffill_shift(s, keys):
    """Within each group: last non-null value strictly before the row."""
    return s.groupby(keys, sort=False).ffill().groupby(keys, sort=False).shift()


def row_state(fb):
    """State of every feedback row's (user, item) pair from strictly earlier events.
    Returns fb (sorted) with columns used by features()."""
    fb = fb.sort_values(["user_id", "timestamp", "event_id"], kind="stable").reset_index(drop=True)
    # user-level: clicks before this event, last clicked item before this event
    ev = fb.groupby("event_id", sort=False).agg(user_id=("user_id", "first"), ts=("timestamp", "first"),
                                               clk=("clicked", "max"))
    ci = fb[fb["clicked"] == 1].drop_duplicates("event_id").set_index("event_id")["itemID"]
    ev["citem"] = ci.reindex(ev.index)
    ev = ev.sort_values(["user_id", "ts"], kind="stable")
    g = ev.groupby("user_id", sort=False)
    ev["u_clicks_before"] = g["clk"].cumsum() - ev["clk"]
    ev["last_item"] = _ffill_shift(ev["citem"], ev["user_id"])
    fb = fb.join(ev[["u_clicks_before", "last_item"]], on="event_id")

    fb = fb.sort_values(["user_id", "itemID", "timestamp", "event_id"], kind="stable").reset_index(drop=True)
    gi = fb.groupby(["user_id", "itemID"], sort=False)
    c = fb["clicked"].to_numpy(float)
    t = fb["timestamp"].to_numpy(float)
    fb["n_imp"] = gi.cumcount()
    fb["n_clk"] = gi["clicked"].cumsum() - fb["clicked"]
    lc = fb["timestamp"].where(fb["clicked"] == 1)
    fb["last_clk"] = _ffill_shift(lc, [fb["user_id"], fb["itemID"]])
    fb["last_shown"] = gi["timestamp"].shift()
    # user click count right after the item's last click -> clicks since
    uc = (fb["u_clicks_before"] + 1).where(fb["clicked"] == 1)
    fb["uc_at_last"] = _ffill_shift(uc, [fb["user_id"], fb["itemID"]])
    # impressions since last click: rows since last click row in the group
    blk = gi["clicked"].cumsum() - fb["clicked"]            # number of clicks before -> block id
    fb["imp_since_clk"] = fb.groupby([fb["user_id"], fb["itemID"], blk], sort=False).cumcount()
    t0 = t.min()
    for name, h in HALF.items():
        e = np.exp((t - t0) / h)
        s = pd.Series(c * e).groupby([fb["user_id"], fb["itemID"]], sort=False).cumsum().to_numpy() - c * e
        fb[name] = s / e
    return fb


def end_state(fb):
    """Final (user, item) state after all of fb, for prediction at later times."""
    fb = fb.sort_values(["user_id", "timestamp", "event_id"], kind="stable")
    ev = fb.groupby("event_id", sort=False).agg(user_id=("user_id", "first"), ts=("timestamp", "first"),
                                               clk=("clicked", "max")).sort_values(["user_id", "ts"])
    ev["u_after"] = ev.groupby("user_id")["clk"].cumsum()          # user clicks incl. this event
    fb = fb.join(ev[["u_after"]], on="event_id")
    t0 = float(fb["timestamp"].min())
    tend = float(fb["timestamp"].max())
    d = {"n_imp": ("clicked", "size"), "n_clk": ("clicked", "sum"), "last_shown": ("timestamp", "max")}
    st = fb.groupby(["user_id", "itemID"]).agg(**d)
    cl = fb[fb["clicked"] == 1]
    st["last_clk"] = cl.groupby(["user_id", "itemID"])["timestamp"].max()
    st["uc_at_last"] = cl.groupby(["user_id", "itemID"])["u_after"].max()
    # impressions after the last click
    m = fb.join(st["last_clk"], on=["user_id", "itemID"])
    st["imp_since_clk"] = m[m["timestamp"] > m["last_clk"]].groupby(["user_id", "itemID"]).size()
    st["imp_since_clk"] = st["imp_since_clk"].fillna(0)
    for name, h in HALF.items():
        w = np.exp((cl["timestamp"].to_numpy(float) - tend) / h)
        st[name] = pd.Series(w, index=cl.index).groupby([cl["user_id"], cl["itemID"]]).sum()
    st[list(HALF)] = st[list(HALF)].fillna(0.0)
    st = st.reset_index()
    u = ev.groupby("user_id").agg(u_clicks=("clk", "sum"))
    last = fb[fb["clicked"] == 1].sort_values("timestamp").drop_duplicates("user_id", keep="last")
    u["last_item"] = last.set_index("user_id")["itemID"]
    return st, u, tend


def transitions(fb):
    """Counts of (previous clicked item -> next clicked item) over users' click sequences."""
    cl = fb[fb["clicked"] == 1].sort_values(["user_id", "timestamp"], kind="stable")
    prev = cl.groupby("user_id")["itemID"].shift()
    tr = pd.DataFrame({"a": prev, "b": cl["itemID"]}).dropna()
    tr = tr[tr["a"] != tr["b"]]
    return tr.groupby(["a", "b"]).size()


# --------------------------------------------------------------- features
def features(df, appeal, trans, loo=None):
    """df needs: itemID, t (request time), n_imp, n_clk, last_clk, last_shown, uc_at_last,
    u_clicks (user clicks before t), imp_since_clk, dec6h, dec2d (valid at t), last_item."""
    n_imp = df["n_imp"].fillna(0).to_numpy(float)
    n_clk = df["n_clk"].fillna(0).to_numpy(float)
    clicked = (n_clk > 0).astype(float)
    t = df["t"].to_numpy(float)
    hsc = np.where(clicked > 0, (t - df["last_clk"].fillna(0).to_numpy(float)) / H, 0.0)
    hss = np.where(n_imp > 0, (t - df["last_shown"].fillna(0).to_numpy(float)) / H, 0.0)
    since = np.where(clicked > 0, df["u_clicks"].fillna(0).to_numpy(float) - df["uc_at_last"].fillna(0).to_numpy(float), 0.0)
    isc = df["imp_since_clk"].fillna(0).to_numpy(float)
    never = (clicked == 0) & (n_imp > 0)
    key = pd.MultiIndex.from_arrays([df["last_item"].astype(object).to_numpy(), df["itemID"].to_numpy()])
    tr = trans.reindex(key).fillna(0).to_numpy(float) if len(trans) else np.zeros(len(df))
    if loo is not None:
        tr = np.maximum(tr - loo, 0)
    X = np.column_stack([
        df["itemID"].map(appeal).fillna(appeal.min() if len(appeal) else 0).to_numpy(float),
        clicked,
        np.log1p(n_clk),
        df["dec6h"].fillna(0).to_numpy(float),
        df["dec2d"].fillna(0).to_numpy(float),
        np.log1p(np.maximum(hsc, 0)),
        np.log1p(np.maximum(since, 0)),
        ((clicked > 0) & (df["itemID"].to_numpy() == df["last_item"].astype(object).to_numpy())).astype(float),
        (clicked > 0) & (isc == 0), (clicked > 0) & (isc == 1), (clicked > 0) & (isc == 2),
        (clicked > 0) & (isc >= 3) & (isc <= 5), (clicked > 0) & (isc >= 6),
        never & (n_imp == 1), never & (n_imp == 2), never & (n_imp >= 3) & (n_imp <= 4),
        never & (n_imp >= 5) & (n_imp <= 9), never & (n_imp >= 10),
        np.log1p(np.maximum(hss, 0)),
        np.log1p(tr),
    ]).astype(float)
    return X


def train_rows(fb, trans, min_ts=None):
    """Slate rows of clicked events (>=2 items, exactly one click) with their history state."""
    st = row_state(fb)
    st = st.rename(columns={"timestamp": "t", "u_clicks_before": "u_clicks"})
    size = st.groupby("event_id")["itemID"].transform("size")
    pos = st.groupby("event_id")["clicked"].transform("sum")
    st = st[(size >= 2) & (pos == 1)]
    if min_ts is not None:
        st = st[st["t"] >= min_ts]
    loo = ((st["clicked"] == 1) & (st["last_item"].astype(object) != st["itemID"]) & st["last_item"].notna()).to_numpy(float)
    return st, loo


def fit_logit(X, ev, y, l2=L2):
    from scipy.optimize import minimize
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    E = ev.max() + 1

    def loss(w):
        u = Z @ w
        mx = np.full(E, -np.inf); np.maximum.at(mx, ev, u)
        e = np.exp(u - mx[ev]); s = np.bincount(ev, weights=e, minlength=E)
        p = e / s[ev]
        f = -(y * (u - mx[ev] - np.log(s[ev]))).sum() + 0.5 * l2 * (w @ w)
        return f, Z.T @ (p - y) + l2 * w

    res = minimize(loss, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x, mu, sd


def pred_frame(rows, cands, st, u, tend):
    """rows: rowid,user_id,timestamp ; cands: rowid,itemID -> frame ready for features()."""
    df = cands.merge(rows[["rowid", "user_id", "timestamp"]], on="rowid")
    df = df.merge(st, on=["user_id", "itemID"], how="left")
    df = df.join(u, on="user_id")
    df["t"] = float(tend) if CLOCK_AT_END else df["timestamp"]   # live check always uses df["timestamp"]
    gap = (df["t"].to_numpy(float) - tend)
    for name, h in HALF.items():
        df[name] = df[name].fillna(0) * np.exp(-np.maximum(gap, 0) / h)
    return df



def pooled_start(fb):
    """Start of the first pooled-slate day (avg slate size > POOLED_MIN_SLATE), or None."""
    e = fb.groupby("event_id").agg(t=("timestamp", "first"), n=("itemID", "size"))
    e["day"] = e["t"] // DAY
    d = e.groupby("day").agg(n=("n", "mean"), k=("n", "size"))
    pooled = d[d["n"] > POOLED_MIN_SLATE]
    return int(pooled.index.min() * DAY) if len(pooled) else None


def train_set(fb, trans):
    tr, loo = train_rows(fb, trans)
    if POOLED_ONLY:
        p = pooled_start(fb)
        if p is not None:
            keep = (tr["t"] >= p).to_numpy()
            if tr.loc[keep, "event_id"].nunique() >= MIN_CHOOSER_EVENTS:
                tr, loo = tr[keep], loo[keep]
    return tr, loo


def seq_picks(fb, rows, cat, appeal):
    """rowid -> best live item according to the sequence-aware choice model."""
    trans = transitions(fb)
    tr, loo = train_set(fb, trans)
    ev, _ = pd.factorize(tr["event_id"])
    print(f"seq model training: {len(tr)} slate rows, {ev.max() + 1} events")
    if ev.max() + 1 < MIN_CHOOSER_EVENTS:
        return None
    w, mu, sd = fit_logit(features(tr, appeal, trans, loo), ev, tr["clicked"].to_numpy(float))
    print("seq weights:", ", ".join(f"{f}={v:+.2f}" for f, v in zip(FEATS, w)))
    st, u, tend = end_state(fb)
    if SEQ_CANDIDATES == "clicked":
        st_c = st[st["n_clk"] > 0][["user_id", "itemID"]]
    else:
        st_c = st[["user_id", "itemID"]]
    live = cat.set_index("itemID")[["valid_from", "valid_to"]]
    glob = appeal.sort_values(ascending=False, kind="stable").index[:N_GLOBAL]
    tt = trans.rename("n").reset_index().sort_values("n", ascending=False, kind="stable").groupby("a").head(N_TRANS)
    out = {}
    rr = rows[["rowid", "user_id", "timestamp"]]
    for s in range(0, len(rr), 5000):
        r = rr.iloc[s:s + 5000]
        parts = [r[["rowid", "user_id"]].merge(st_c, on="user_id")[["rowid", "itemID"]]]
        if SEQ_CANDIDATES != "clicked":
            parts.append(pd.DataFrame({"rowid": np.repeat(r["rowid"].to_numpy(), len(glob)),
                                       "itemID": np.tile(np.asarray(glob), len(r))}))
            lt = r.join(u["last_item"], on="user_id").dropna(subset=["last_item"])
            parts.append(lt.merge(tt, left_on="last_item", right_on="a")[["rowid", "b"]].rename(columns={"b": "itemID"}))
        cand = pd.concat(parts).drop_duplicates()
        df = pred_frame(r, cand, st, u, tend)
        vf, vt = df["itemID"].map(live["valid_from"]), df["itemID"].map(live["valid_to"])
        df = df[vf.notna() & (vf <= df["timestamp"]) & (df["timestamp"] < vt)]
        if df.empty:
            continue
        df = df.assign(s=((features(df, appeal, trans) - mu) / sd) @ w)
        best = df.sort_values(["s", "itemID"], kind="stable").drop_duplicates("rowid", keep="last")
        out.update(zip(best["rowid"], best["itemID"]))
    return out


# ------------------------------------------------------------------ main logic
def recommend(fb, rows, cat):
    """Returns list of (rowid, itemID) for every row. rows: rowid, user_id, timestamp."""
    items = cat["itemID"].to_numpy()
    vf = cat["valid_from"].to_numpy(dtype="int64")
    vt = cat["valid_to"].to_numpy(dtype="int64")
    live = cat.set_index("itemID")[["valid_from", "valid_to"]]

    try:
        score = global_scores(fb, items)
    except Exception as e:
        print("global scoring failed, catalogue order:", e)
        score = np.zeros(len(items))
    appeal = pd.Series(score, index=items)

    upick = None
    if PERSONAL_RANK == "seq":
        try:
            upick = seq_picks(fb, rows, cat, appeal)
        except Exception as e:
            print("seq model failed, using v3 rules:", e)
    rank = "learned" if (PERSONAL_RANK == "seq" and upick is None) else PERSONAL_RANK
    chooser = None
    if rank == "learned":
        try:
            chooser = fit_chooser(fb, appeal)
        except Exception as e:
            print("chooser failed, using most recent:", e)
    if upick is None:
        try:
            upick = personal_picks(fb, rows, live, appeal, chooser, rank)
        except Exception as e:
            print("personal picks failed:", e)
            upick = {}
    try:
        rej = rejected_pairs(fb) if SKIP_REJECTED else set()
    except Exception as e:
        print("rejected pairs failed:", e)
        rej = set()
    print(f"personal picks: {len(upick)} of {len(rows)} rows")

    order = np.argsort(-score, kind="stable")
    items_s, vf_s, vt_s = items[order], vf[order], vt[order]
    k = min(TOP_K, len(items_s))
    rowids = rows["rowid"].to_numpy()
    users = rows["user_id"].to_numpy()
    ts = rows["timestamp"].to_numpy(dtype="int64")

    for start in range(0, len(rowids), 256):
        r, u = rowids[start:start + 256], users[start:start + 256]
        t = ts[start:start + 256][:, None]
        ok = (vf_s[None, :k] <= t) & (t < vt_s[None, :k])
        out = []
        for j in range(len(r)):
            if r[j] in upick:
                out.append((str(r[j]), str(upick[r[j]])))
                continue
            cand = np.flatnonzero(ok[j])
            if len(cand) == 0:                                  # nothing live in top k: full scan
                cand = np.flatnonzero((vf_s <= t[j, 0]) & (t[j, 0] < vt_s))
            pick = None
            for c in cand:
                if (u[j], items_s[c]) not in rej:
                    pick = items_s[c]
                    break
            if pick is None:
                pick = items_s[cand[0]] if len(cand) else items_s[0]
            out.append((str(r[j]), str(pick)))
        yield out


def predict(data_dir, batch_size=256):
    data = Path(data_dir)
    rows = read_any(data, "x_pred", dtype={"rowid": str, "user_id": str})
    rows = rows.astype({"rowid": str, "user_id": str}).drop_duplicates("rowid")
    rows["timestamp"] = rows["timestamp"].astype("int64")
    cat = read_any(data, "catalog", dtype={"itemID": str}).astype({"itemID": str}).drop_duplicates("itemID")
    try:
        fb = load_feedback(data)
    except Exception as e:
        print("feedback failed:", e)
        fb = pd.DataFrame(columns=["event_id", "user_id", "timestamp", "itemID", "clicked"])
    batch = []
    for part in recommend(fb, rows, cat):
        batch.extend(part)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch