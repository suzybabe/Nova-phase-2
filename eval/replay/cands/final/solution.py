"""Team Nova - Phase 2 hybrid "cfv4_post" (referee; not a submission).

v4c exactly, plus cold_post's post-expiry rule for UNSEEN cold users only:
- warm rows (live prior click): cf_v3 rank, as in v4c
- SEEN cold users (in feedback, no live prior): v3 pick at all times, if live, as in v4c
- UNSEEN cold users (no feedback at all): v2 fallback while itm_26ec3536 is live;
  after it expires, the best live item by v3's pooled-slate appeal (cold_post rule)
Safety guards of v4c are kept.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

USE_V3_MULTI = False
HALF_LIFE_DAYS = 1.0
PRIOR_STRENGTH = 100.0
LAM = 5.0
LAM_U = 0.3
TIME_BINS = 24
W_MNL = 0.7
TOP_K = 512
DAY = 86400
RECLICK_HOURS = 1e9
RANK = "count"          # "count" = most clicks, tie recent; "recent" = HIST=0
BUDGET_S = 150.0
import time as _time
_T0 = [_time.time()]


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
        return read_any(data, "feedback_long", dtype={"itemID": str, "user_id": str, "event_id": str})
    except Exception as e:
        print("no feedback_long, reading jsonl:", e)
    rows = []
    with open(data / "feedback.jsonl") as f:
        for line in f:
            e = json.loads(line)
            for it in e["displayed_items"]:
                rows.append((e["event_id"], e["user_id"], e["timestamp"], it, int(it == e.get("clicked_item"))))
    return pd.DataFrame(rows, columns=["event_id", "user_id", "timestamp", "itemID", "clicked"])


def smoothed_ctr(fb, items):
    ts = fb["timestamp"].astype("int64").to_numpy()
    w = 0.5 ** ((ts.max() - ts) / (HALF_LIFE_DAYS * DAY))
    g = pd.DataFrame({"itemID": fb["itemID"].to_numpy(), "w": w,
                      "wc": w * fb["clicked"].astype(float).to_numpy()}).groupby("itemID")[["w", "wc"]].sum()
    g_ctr = fb["clicked"].astype(float).mean()
    ctr = (g["wc"] + PRIOR_STRENGTH * g_ctr) / (g["w"] + PRIOR_STRENGTH)
    return ctr.reindex(items).fillna(g_ctr).to_numpy()


def fit_mnl(fb):
    from scipy.optimize import minimize
    fb = fb.sort_values("event_id").reset_index(drop=True)
    ev, _ = pd.factorize(fb["event_id"])
    it, item_ids = pd.factorize(fb["itemID"])
    E, I = ev.max() + 1, len(item_ids)
    clk = fb["clicked"].astype(float).to_numpy()
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


def logger_slice(fb):
    # the random logger shows 3 items; later official feedback is pooled team slates (4-6 items)
    # and a plain item MNL on those slates credits habit re-clicks to the item. fit appeal on the
    # logger days only (mean slate length <= 3.5). fall back to all events if that looks too thin.
    try:
        n = fb.groupby("event_id").size()
        ts = fb.groupby("event_id")["timestamp"].first()
        day = ts // 86400
        mean_n = n.groupby(day).mean()
        good = mean_n[mean_n <= 3.5].index
        ev = day[day.isin(good)].index
        out = fb[fb["event_id"].isin(ev)]
        if out["event_id"].nunique() >= 1000:
            print("item scores on logger-like days:", int(out["event_id"].nunique()), "events (skipped pooled slates)")
            return out
    except Exception as e:
        print("logger slice failed, using all events:", e)
    return fb


def item_scores(fb, items):
    fb = fb.astype({"itemID": str, "user_id": str, "event_id": str})
    fb_fit = logger_slice(fb)
    ctr = smoothed_ctr(fb_fit, items)
    try:
        b = fit_mnl(fb_fit).reindex(items).fillna(0.0).to_numpy()
        s = W_MNL * b + (1 - W_MNL) * np.log(ctr)
        if not np.isfinite(s).all():
            raise ValueError("non finite mnl scores")
        return s
    except Exception as e:
        print("mnl failed, using plain ctr:", e)
        return ctr


W_CTR, W_NHL, HL_H, BETA = 0.5, 4.0, 120.0, 1.0


def habit_feats(clk_times, users, items, ts):
    # same 4 habit terms as the scorer: prior click, log1p(count), log1p(hours since last), last < 12h
    F = np.zeros((len(users), 4), np.float32)
    for r, (u, i, t) in enumerate(zip(users, items, ts)):
        a = clk_times.get((u, i))
        if a is None:
            continue
        k = np.searchsorted(a, t)
        if k == 0:
            continue
        dt = (t - a[k - 1]) / 3600.0
        F[r] = (1.0, np.log1p(k), np.log1p(dt), float(dt < 12))
    return F


def fit_pooled_appeal(fb, lam=1.0, iters=300):
    # conditional logit on clicked pooled slates: u = b_item + w . habit ; returns pd.Series b
    import torch
    n = fb.groupby("event_id").size()
    ts = fb.groupby("event_id")["timestamp"].first()
    day = ts // 86400
    mean_n = n.groupby(day).mean()
    pooled_days = mean_n[mean_n > 3.5].index
    evs = day[day.isin(pooled_days)].index
    clicked = fb[fb["clicked"].astype(int) == 1]
    evs = evs[evs.isin(clicked["event_id"])]
    if len(evs) < 1000:
        print("not enough pooled slates for b_pooled:", len(evs))
        return None
    lg = fb[fb["event_id"].isin(evs)].sort_values(["event_id", "itemID"]).reset_index(drop=True)
    c = clicked.sort_values("timestamp")
    clk_times = {k: v["timestamp"].to_numpy() for k, v in c.groupby(["user_id", "itemID"])}
    F = habit_feats(clk_times, lg["user_id"].to_numpy(), lg["itemID"].to_numpy(), lg["timestamp"].to_numpy())
    e = pd.factorize(lg["event_id"])[0]
    k = lg.groupby(e).cumcount().to_numpy()
    items = pd.Index(lg["itemID"].unique())
    E, K = e.max() + 1, k.max() + 1
    IT = np.zeros((E, K), np.int64); M = np.zeros((E, K), bool); X = np.zeros((E, K, 4), np.float32); Y = np.zeros((E, K), np.float32)
    IT[e, k] = items.get_indexer(lg["itemID"]); M[e, k] = True; X[e, k] = F; Y[e, k] = lg["clicked"].astype(float).to_numpy()
    IT, M, X, Y = map(torch.tensor, (IT, M, X, Y))
    b = torch.zeros(len(items), requires_grad=True)
    w = torch.zeros(4, requires_grad=True)
    opt = torch.optim.LBFGS([b, w], max_iter=iters, line_search_fn="strong_wolfe")

    def clos():
        opt.zero_grad()
        s = torch.where(M, b[IT] + (X * w).sum(-1), torch.tensor(-1e4))
        loss = -(torch.log_softmax(s, 1) * Y).sum() / E + lam * (b ** 2).sum() / E
        loss.backward()
        return loss
    opt.step(clos)
    bb = pd.Series(b.detach().numpy().astype(float), index=items)
    print("b_pooled fitted on", E, "pooled slates, habit w", np.round(w.detach().numpy(), 3))
    return bb


def user_picks(fb, rows, vf_map, vt_map, ctr_map=None, g_ctr=None):
    # rowid -> a live item the user clicked before (any lag), ranked as in the docstring
    try:
        fb = fb.astype({"user_id": str, "itemID": str})
        clk = fb[fb["clicked"].astype(int) == 1][["user_id", "itemID", "timestamp"]].rename(columns={"timestamp": "tc"})
        m = rows[["rowid", "user_id", "timestamp"]].merge(clk, on="user_id")
        gap = m["timestamp"].astype("int64") - m["tc"].astype("int64")
        m = m[(gap >= 0) & (gap <= RECLICK_HOURS * 3600)].copy()
        m["d"] = 0.5 ** ((m["timestamp"].astype("int64") - m["tc"].astype("int64")) / (HL_H * 3600.0))
        m = m.groupby(["rowid", "itemID"]).agg(nclk=("tc", "size"), tc=("tc", "max"), nhl=("d", "sum"),
                                                timestamp=("timestamp", "first")).reset_index()
        m["vf"] = m["itemID"].map(vf_map)
        m["vt"] = m["itemID"].map(vt_map)
        t = m["timestamp"].astype("int64")
        m = m[m["vf"].notna() & (m["vf"] <= t) & (t < m["vt"])].copy()
        _NLIVE.clear(); _NLIVE.update(m.groupby("rowid").size().to_dict())
        r_m2 = m["nclk"].astype(float) * 1e12 + m["tc"].astype(float)
        try:
            if _time.time() - _T0[0] > BUDGET_S or ctr_map is None:
                raise RuntimeError("over budget or no ctr")
            c = m["itemID"].map(ctr_map).astype(float).fillna(g_ctr)
            key = W_CTR * np.log(np.clip(c, 1e-6, 1.0)) + W_NHL * m["nhl"]
            zb = 0.0
            try:
                if _time.time() - _T0[0] < BUDGET_S * 0.6:
                    bb = fit_pooled_appeal(fb)
                    if bb is not None:
                        b = m["itemID"].map(bb).fillna(float(np.percentile(bb.to_numpy(), 25)))
                        zb = (b - b.mean()) / (b.std() + 1e-9)
            except Exception as e:
                print("b_pooled failed, beta term off:", e)
            m["r"] = key + BETA * zb + m["tc"].astype(float) / 1e12
            if not np.isfinite(m["r"]).all():
                raise ValueError("non finite rank")
        except Exception as e:
            print("cf_v3 rank failed, model_angle2 rank:", e)
            m["r"] = r_m2
        m = m.sort_values("r").drop_duplicates("rowid", keep="last")
        return dict(zip(m["rowid"], m["itemID"]))
    except Exception as e:
        print("user picks failed, global only:", e)
        return {}


def v2_predict(data_dir, batch_size=256, _info=None):
    _T0[0] = _time.time()
    data = Path(data_dir)
    rows = read_any(data, "x_pred", dtype={"rowid": str, "user_id": str})
    rows = rows.astype({"rowid": str, "user_id": str}).drop_duplicates("rowid")
    cat = read_any(data, "catalog", dtype={"itemID": str}).astype({"itemID": str}).drop_duplicates("itemID")

    items = cat["itemID"].to_numpy()
    vf = cat["valid_from"].to_numpy(dtype="int64")
    vt = cat["valid_to"].to_numpy(dtype="int64")

    fb = None
    try:
        fb = load_feedback(data)
        score = item_scores(fb, items)
    except Exception as e:
        print("scoring failed, using catalogue order:", e)
        score = np.zeros(len(items))
    upick = {}
    if fb is not None:
        ctr_map, g_ctr = None, None
        try:
            fbl = logger_slice(fb.astype({"itemID": str, "user_id": str, "event_id": str}))
            ctr_map = pd.Series(smoothed_ctr(fbl, items), index=items)
            g_ctr = float(fbl["clicked"].astype(float).mean())
        except Exception as e:
            print("logger ctr failed:", e)
        upick = user_picks(fb, rows, pd.Series(vf, index=items), pd.Series(vt, index=items), ctr_map, g_ctr)
    if _info is not None:
        _info["upick"] = upick
        _info["fb_users"] = set(fb["user_id"].astype(str)) if fb is not None else set()
    print("user picks:", len(upick), "rank: cf_v3", "elapsed", f"{_time.time()-_T0[0]:.1f}s")

    order = np.argsort(-score, kind="stable")
    items_s, vf_s, vt_s = items[order], vf[order], vt[order]
    k = min(TOP_K, len(items_s))

    rowids = rows["rowid"].to_numpy()
    ts = rows["timestamp"].to_numpy(dtype="int64")
    for start in range(0, len(rowids), batch_size):
        r = rowids[start:start + batch_size]
        t = ts[start:start + batch_size][:, None]
        ok = (vf_s[None, :k] <= t) & (t < vt_s[None, :k])
        has = ok.any(1)
        chosen = items_s[np.where(has, ok.argmax(1), 0)].copy()
        for j in np.where(~has)[0]:
            full = np.where((vf_s <= t[j, 0]) & (t[j, 0] < vt_s))[0]
            chosen[j] = items_s[full[0]] if len(full) else items_s[0]
        out = []
        for a, b in zip(r, chosen):
            out.append((str(a), str(upick.get(a, b))))
        yield out


V3_LAM = 1.0
V3_B_PCT = 25
V3_TOP_K = 256


def v3_load_events(data):
    # event level: event_id, user_id, timestamp, displayed_items (list), clicked_item (None = no click)
    try:
        rows = []
        with open(data / "feedback.jsonl") as f:
            for line in f:
                e = json.loads(line)
                rows.append((str(e["event_id"]), str(e["user_id"]), int(e["timestamp"]),
                             [str(x) for x in e["displayed_items"]], e.get("clicked_item")))
        return pd.DataFrame(rows, columns=["event_id", "user_id", "timestamp", "displayed_items", "clicked_item"])
    except Exception as e:
        print("jsonl failed, using long format:", e)
    fl = read_any(data, "feedback_long", dtype={"itemID": str, "user_id": str, "event_id": str})
    fl = fl.astype({"itemID": str, "user_id": str, "event_id": str})
    g = fl.groupby("event_id", sort=False)
    ev = g.agg(user_id=("user_id", "first"), timestamp=("timestamp", "first"), displayed_items=("itemID", list)).reset_index()
    ck = fl[fl["clicked"].astype(int) == 1].drop_duplicates("event_id").set_index("event_id")["itemID"]
    ev["clicked_item"] = ev["event_id"].map(ck)
    return ev


def v3_habit(q, clicks):
    # q: DataFrame key,user_id,itemID,timestamp -> (n,4) v3_habit features from clicks strictly before timestamp
    F = np.zeros((len(q), 4))
    m = q[["key", "user_id", "itemID", "timestamp"]].merge(clicks, on=["user_id", "itemID"])
    m = m[m["tc"] < m["timestamp"]]
    if len(m):
        a = m.groupby("key").agg(n=("tc", "size"), last=("tc", "max"), t=("timestamp", "first"))
        pos = pd.Series(np.arange(len(q)), index=q["key"].to_numpy()).reindex(a.index).to_numpy()
        dt = (a["t"].to_numpy() - a["last"].to_numpy()) / 3600.0
        F[pos, 0] = 1.0
        F[pos, 1] = np.log1p(a["n"].to_numpy())
        F[pos, 2] = np.log1p(dt)
        F[pos, 3] = (dt < 12).astype(float)
    return F


def v3_fit_choice(ev, clicks):
    import torch
    ce = ev[ev["clicked_item"].notna()]
    lg = ce[["event_id", "user_id", "timestamp", "displayed_items", "clicked_item"]].explode("displayed_items")
    lg = lg.rename(columns={"displayed_items": "itemID"}).reset_index(drop=True)
    lg["key"] = np.arange(len(lg))
    X = v3_habit(lg, clicks)
    items = pd.Index(lg["itemID"].unique())
    e = pd.factorize(lg["event_id"])[0]
    k = lg.groupby(e).cumcount().to_numpy()
    E, K = e.max() + 1, k.max() + 1
    IT = np.zeros((E, K), np.int64); M = np.zeros((E, K), bool); XX = np.zeros((E, K, 4), np.float32); Y = np.zeros((E, K), np.float32)
    IT[e, k] = items.get_indexer(lg["itemID"]); M[e, k] = True; XX[e, k] = X
    Y[e, k] = (lg["itemID"] == lg["clicked_item"]).to_numpy()
    IT, M, XX, Y = map(torch.tensor, (IT, M, XX, Y))
    b = torch.zeros(len(items), dtype=torch.float32, requires_grad=True)
    w = torch.zeros(4, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([b, w], max_iter=300, line_search_fn="strong_wolfe")
    neg = torch.tensor(-1e4)

    def clos():
        opt.zero_grad()
        s = torch.where(M, b[IT] + (XX * w).sum(-1), neg)
        loss = -(torch.log_softmax(s, 1) * Y).sum() / E + V3_LAM * (b ** 2).sum() / E
        loss.backward()
        return loss
    opt.step(clos)
    bb = b.detach().numpy().astype(float)
    if not np.isfinite(bb).all():
        raise ValueError("non finite b")
    print("choice fit: v3_habit w", np.round(w.detach().numpy(), 3), "items", len(items), "events", E)
    return pd.Series(bb, index=items), w.detach().numpy().astype(float)


def v3_predict(data_dir, batch_size=256):
    data = Path(data_dir)
    rows = read_any(data, "x_pred", dtype={"rowid": str, "user_id": str})
    rows = rows.astype({"rowid": str, "user_id": str}).drop_duplicates("rowid").reset_index(drop=True)
    rows["timestamp"] = rows["timestamp"].astype("int64")
    cat = read_any(data, "catalog", dtype={"itemID": str}).astype({"itemID": str}).drop_duplicates("itemID")
    items = cat["itemID"].to_numpy()
    vf = cat["valid_from"].to_numpy(dtype="int64"); vt = cat["valid_to"].to_numpy(dtype="int64")
    vfm = pd.Series(vf, index=items); vtm = pd.Series(vt, index=items)

    clicks = pd.DataFrame(columns=["user_id", "itemID", "tc"])
    bser, w = None, np.zeros(4)
    try:
        ev = v3_load_events(data)
        c = ev[ev["clicked_item"].notna()]
        clicks = pd.DataFrame({"user_id": c["user_id"].astype(str).to_numpy(), "itemID": c["clicked_item"].astype(str).to_numpy(),
                               "tc": c["timestamp"].astype("int64").to_numpy()})
        bser, w = v3_fit_choice(ev, clicks)
    except Exception as e:
        print("choice fit failed:", e)
    if bser is None:
        # fallback appeal: click counts
        bser = clicks.groupby("itemID").size().astype(float) if len(clicks) else pd.Series(dtype=float)
        w = np.array([1.0, 1.3, 0.0, 0.0])
    bdef = float(np.percentile(bser.to_numpy(), V3_B_PCT)) if len(bser) else 0.0
    bcat = bser.reindex(items).fillna(bdef).to_numpy()

    # best live item by b for each row (global part)
    order = np.argsort(-bcat, kind="stable")
    items_s, vf_s, vt_s, b_s = items[order], vf[order], vt[order], bcat[order]
    k = min(V3_TOP_K, len(items_s))
    ts = rows["timestamp"].to_numpy()
    gpick = np.empty(len(rows), dtype=object); gu = np.zeros(len(rows))
    for s0 in range(0, len(rows), 4096):
        t = ts[s0:s0 + 4096][:, None]
        ok = (vf_s[None, :k] <= t) & (t < vt_s[None, :k])
        has = ok.any(1); j = np.where(has, ok.argmax(1), 0)
        for r in np.where(~has)[0]:
            full = np.where((vf_s <= t[r, 0]) & (t[r, 0] < vt_s))[0]
            j[r] = full[0] if len(full) else 0
        gpick[s0:s0 + 4096] = items_s[j]; gu[s0:s0 + 4096] = b_s[j]

    # personal part: user's previously clicked live items
    pick = {}
    try:
        q = rows[["rowid", "user_id", "timestamp"]].merge(clicks[["user_id", "itemID"]].drop_duplicates(), on="user_id")
        t = q["timestamp"].to_numpy()
        q = q[(q["itemID"].map(vfm) <= t) & (t < q["itemID"].map(vtm))].reset_index(drop=True)
        q["key"] = np.arange(len(q))
        X = v3_habit(q, clicks)
        q = q[X[:, 0] > 0].copy(); X = X[X[:, 0] > 0]
        q["u"] = bser.reindex(q["itemID"]).fillna(bdef).to_numpy() + X @ w
        q = q.sort_values("u").drop_duplicates("rowid", keep="last")
        gmap = pd.Series(gu, index=rows["rowid"].to_numpy())
        q = q[q["u"].to_numpy() > gmap.reindex(q["rowid"]).to_numpy()]
        pick = dict(zip(q["rowid"], q["itemID"]))
    except Exception as e:
        print("personal picks failed:", e)
    print("personal picks:", len(pick), "of", len(rows))

    rid = rows["rowid"].to_numpy()
    for s0 in range(0, len(rid), batch_size):
        yield [(str(r), str(pick.get(r, g))) for r, g in zip(rid[s0:s0 + batch_size], gpick[s0:s0 + batch_size])]


V3_BUDGET_S = 120.0
_NLIVE = {}


def predict(data_dir, batch_size=256):
    t0 = _time.time()
    info = {}
    base = [x for b in v2_predict(data_dir, batch_size=4096, _info=info) for x in b]
    upick, fbu = info.get("upick", {}), info.get("fb_users", set())
    nlive = dict(_NLIVE)
    out = dict(base)
    try:
        data = Path(data_dir)
        rows = read_any(data, "x_pred", dtype={"rowid": str, "user_id": str}).astype({"rowid": str, "user_id": str})
        rows = rows.drop_duplicates("rowid")
        rows["timestamp"] = rows["timestamp"].astype("int64")
        cat = read_any(data, "catalog", dtype={"itemID": str}).astype({"itemID": str}).drop_duplicates("itemID")
        g = cat[cat["itemID"].astype(str).str.startswith("itm_26ec3536")]
        gtop_vt = int(g["valid_to"].iloc[0]) if len(g) else None
        vf_map = pd.Series(cat["valid_from"].to_numpy(dtype="int64"), index=cat["itemID"])
        vt_map = pd.Series(cat["valid_to"].to_numpy(dtype="int64"), index=cat["itemID"])
        rid = rows["rowid"].to_numpy()
        uid = rows["user_id"].to_numpy()
        ts = rows["timestamp"].to_numpy()
        tgt = set()
        n_seen = n_unseen_post = 0
        for r, u, t in zip(rid, uid, ts):
            if r not in upick and u in fbu:
                tgt.add(r)                       # v4c: seen cold, v3 at all times
                n_seen += 1
            elif r not in upick and u not in fbu and gtop_vt is not None and int(t) >= gtop_vt:
                tgt.add(r)                       # unseen post-expiry: cold_post
                n_unseen_post += 1
            elif USE_V3_MULTI and nlive.get(r, 0) >= 2:
                tgt.add(r)
        print("cfv4_post targets: seen-cold", n_seen, "unseen-post", n_unseen_post, "gtop_vt", gtop_vt)
        if tgt and _time.time() - t0 < V3_BUDGET_S:
            v3 = {}
            for b in v3_predict(data_dir, batch_size=4096):
                for r, it in b:
                    if r in tgt:
                        v3[r] = it
            n = 0
            ts_map = dict(zip(rows["rowid"].astype(str), rows["timestamp"].to_numpy()))
            for r, it in v3.items():
                if not (isinstance(it, str) and it and it not in ("None", "nan")):
                    continue
                t = ts_map.get(str(r))
                if t is None or it not in vf_map.index:
                    continue
                if int(vf_map.loc[it]) <= int(t) < int(vt_map.loc[it]):
                    out[r] = it; n += 1
            print("cfv4_post: rows switched to v3 pick:", n, "of target", len(tgt))
        else:
            print("cfv4_post: v3 skipped (no target rows or over budget)")
    except Exception as e:
        print("cfv4_post: v3 part failed, keeping v4c/v2 picks:", e)
    keys = [r for r, _ in base]
    for s0 in range(0, len(keys), batch_size):
        yield [(r, out[r]) for r in keys[s0:s0 + batch_size]]
