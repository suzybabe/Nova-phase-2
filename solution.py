import os
import numpy as np
import pandas as pd

# baseline: smoothed click-through rate per item, blended overall + last 24h,
# plus a small boost for items the user clicked before. always returns a live item.

HERE = os.path.dirname(os.path.abspath(__file__))
STRENGTH = 20      # smoothing, pulls low-view items towards the average
RECENT_W = 0.3     # weight on the last-24h ctr
USER_BOOST = 0.05  # bump for items this user clicked before


def read(data_dir, name):
    pq = os.path.join(data_dir, name + ".parquet")
    if os.path.exists(pq):
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    return pd.read_csv(os.path.join(data_dir, name + ".csv"))


def item_scores(fb, items):
    prior = fb["clicked"].mean() if len(fb) else 0.0
    g = fb.groupby("itemID")["clicked"].agg(["sum", "count"])
    ctr = (g["sum"] + prior * STRENGTH) / (g["count"] + STRENGTH)

    cutoff = fb["timestamp"].max() - 24 * 3600 if len(fb) else 0
    r = fb[fb["timestamp"] >= cutoff].groupby("itemID")["clicked"].agg(["sum", "count"])
    ctr24 = (r["sum"] + prior * STRENGTH) / (r["count"] + STRENGTH)

    s = pd.Series(prior, index=items)
    s.loc[s.index.intersection(ctr.index)] = ctr.reindex(s.index.intersection(ctr.index)).values
    c24 = pd.Series(prior, index=items)
    c24.loc[c24.index.intersection(ctr24.index)] = ctr24.reindex(c24.index.intersection(ctr24.index)).values
    return ((1 - RECENT_W) * s + RECENT_W * c24).values


def predict(data_dir, batch_size=256):
    req = read(data_dir, "x_pred")
    cat = read(data_dir, "catalog")
    req["timestamp"] = req["timestamp"].astype("int64")

    items = cat["itemID"].astype(str).values
    vf = cat["valid_from"].astype("int64").values
    vt = cat["valid_to"].astype("int64").values

    try:
        fb = read(data_dir, "feedback_long")
        fb["timestamp"] = fb["timestamp"].astype("int64")
        fb["clicked"] = fb["clicked"].astype(float)
        fb = fb.drop_duplicates(subset=["event_id", "itemID"])
        base = item_scores(fb, items)
        clicks = fb[fb["clicked"] == 1]
        user_clicked = clicks.groupby("user_id")["itemID"].agg(lambda x: set(map(str, x))).to_dict()
    except Exception:
        # no usable feedback: fall back to longest-live items first
        base = (vt - vf).astype(float)
        user_clicked = {}

    order = np.argsort(-base, kind="stable")
    items_o, vf_o, vt_o, base_o = items[order], vf[order], vt[order], base[order]
    pos = {it: i for i, it in enumerate(items_o)}

    rowids = req["rowid"].astype(str).values
    users = req["user_id"].astype(str).values
    times = req["timestamp"].values
    seen = set()

    for start in range(0, len(req), batch_size):
        t = times[start:start + batch_size]
        live = (vf_o[None, :] <= t[:, None]) & (t[:, None] < vt_o[None, :])
        score = np.where(live, base_o[None, :], -np.inf)

        for k in range(len(t)):
            uc = user_clicked.get(users[start + k])
            if uc:
                idx = [pos[i] for i in uc if i in pos]
                if idx:
                    score[k, idx] += USER_BOOST * live[k, idx]

        best = score.argmax(axis=1)
        ok = np.isfinite(score[np.arange(len(t)), best])

        batch = []
        for k in range(len(t)):
            rid = rowids[start + k]
            if not ok[k] or rid in seen:
                continue
            seen.add(rid)
            batch.append((rid, items_o[best[k]]))
        if batch:
            yield batch


if __name__ == "__main__":
    import sys, time
    d = sys.argv[1] if len(sys.argv) > 1 else (os.path.expanduser("~/data/round-1/phase-2-round-1-input") if os.path.exists(os.path.expanduser("~/data/round-1/phase-2-round-1-input")) else os.path.expanduser("~/Downloads/phase-2-round-1-input"))
    t0 = time.time()
    n, picked = 0, set()
    for b in predict(d):
        n += len(b)
        picked.update(i for _, i in b)
    print("rows answered:", n, "unique items:", len(picked), "time:", round(time.time() - t0, 1), "s")