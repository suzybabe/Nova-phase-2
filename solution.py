"""solution.py - Team NOVA, Huawei Tech Arena 2026, Phase 2.

One file that joins the three branches:

    PART 1  data-features branch (Darling)  loading, item features, availability
    PART 2  Collaborative-Filtering (Ahmed) ItemItemCF + ImplicitALS, with an ID adapter
    PART 3  model branch (Addy)             ClickRanker: LightGBM ranker on clicked events
    PART 4  predict(data_dir, batch_size)   the function the scorer calls

How it works: everything is trained inside predict() from the round's own data
directory (the feedback there is cumulative), so there is no model file to ship
and the model adapts every round automatically. For each request it builds a
short candidate list (popular available items + the user's CF favourites),
scores them with the ranker and returns the best available item.

If anything fails, or the time budget is nearly used, it falls back to the most
popular available item, so every row always gets a valid answer.

Local check:  python solution.py path/to/round-data
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ------------------------------------------------------------------ settings
USE_CF = True         # use the CF scores as ranker features and for candidates
TIME_BUDGET = 170.0   # seconds; after this the remaining rows use the fast popularity path (limit 240)
CHUNK = 5000          # requests processed at once
TOP_M = 400           # popular items looked at per request
K_POP = 25            # popular available candidates per request
K_CF = 15             # CF candidates per known user
CF_SPLIT = 0.6        # CF for the ranker's training rows is fitted on the oldest 60% of events
CF_FEATURES = ["cf_als", "cf_item"]
DAY = 86400.0


def _log(*a):
    print("[solution]", *a, file=sys.stderr, flush=True)


# =============================================================================
# PART 1 - data & features (data-features branch)
# =============================================================================
def read_table(data_dir, name):
    # parquet is faster, fall back to csv if it's missing or unreadable
    pq = os.path.join(data_dir, name + ".parquet")
    if os.path.exists(pq):
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    return pd.read_csv(os.path.join(data_dir, name + ".csv"))


def load_data(data_dir):
    requests = read_table(data_dir, "x_pred")
    catalog = read_table(data_dir, "catalog")
    feedback = read_table(data_dir, "feedback_long")

    requests["timestamp"] = requests["timestamp"].astype("int64")
    catalog["valid_from"] = catalog["valid_from"].astype("int64")
    catalog["valid_to"] = catalog["valid_to"].astype("int64")
    feedback["timestamp"] = feedback["timestamp"].astype("int64")
    feedback["clicked"] = feedback["clicked"].astype("int8")

    # feedback is cumulative, so drop any repeated rows
    feedback = feedback.drop_duplicates(subset=["event_id", "itemID"])
    requests = requests.drop_duplicates(subset=["rowid"]).reset_index(drop=True)
    catalog = catalog.drop_duplicates(subset=["itemID"]).reset_index(drop=True)
    return requests, catalog, feedback


def smoothed_ctr(clicks, shown, prior, strength=20):
    # pulls items with few views towards the average so they don't get a fake 100%
    return (clicks + prior * strength) / (shown + strength)


def item_features(feedback, now=None):
    if now is None:
        now = feedback["timestamp"].max()
    prior = feedback["clicked"].mean()

    g = feedback.groupby("itemID")
    feats = pd.DataFrame({
        "shown": g.size(),
        "clicks": g["clicked"].sum(),
        "first_seen": g["timestamp"].min(),
        "last_seen": g["timestamp"].max(),
    })
    feats["ctr"] = smoothed_ctr(feats["clicks"], feats["shown"], prior)

    for hours in [24, 6]:
        recent = feedback[feedback["timestamp"] >= now - hours * 3600]
        rg = recent.groupby("itemID")
        feats[f"shown_{hours}h"] = rg.size()
        feats[f"clicks_{hours}h"] = rg["clicked"].sum()
    feats = feats.fillna(0)
    for hours in [24, 6]:
        feats[f"ctr_{hours}h"] = smoothed_ctr(feats[f"clicks_{hours}h"], feats[f"shown_{hours}h"], prior)

    feats["hours_since_seen"] = (now - feats["last_seen"]) / 3600
    return feats.reset_index()


def user_item_clicks(feedback):
    # how many times each user clicked each item, for the CF models
    clicks = feedback[feedback["clicked"] == 1]
    return clicks.groupby(["user_id", "itemID"]).size().rename("n").reset_index()


def pick_best_available(requests, catalog, item_scores, chunk=2000):
    # for each request, pick the highest scoring item that's live at its timestamp
    # item_scores: dict or Series itemID -> score (items with no score get -inf)
    scores = catalog["itemID"].map(item_scores).astype(float).fillna(-np.inf).to_numpy()

    order = np.argsort(-scores, kind="stable")       # best first
    item_ids = catalog["itemID"].to_numpy(dtype=object)[order]
    vf = catalog["valid_from"].to_numpy()[order]
    vt = catalog["valid_to"].to_numpy()[order]

    times = requests["timestamp"].to_numpy()
    picks = np.empty(len(requests), dtype=object)
    for start in range(0, len(requests), chunk):
        t = times[start:start + chunk][:, None]
        live = (vf[None, :] <= t) & (t < vt[None, :])
        first = live.argmax(axis=1)
        chosen = item_ids[first].copy()
        chosen[~live.any(axis=1)] = None
        picks[start:start + chunk] = chosen
    return pd.DataFrame({"rowid": requests["rowid"].to_numpy(dtype=object), "itemID": picks})


# =============================================================================
# PART 2 - collaborative filtering (Collaborative-Filtering branch)
# =============================================================================
class ItemItemCF:
    """Items are similar if the same users clicked both (cosine similarity)."""
    def fit(self, R):
        B = (R > 0).astype(np.float32).tocsc()
        norms = np.sqrt(np.asarray(B.sum(0)).ravel()) + 1e-9
        co = (B.T @ B).toarray()
        self.sim = co / norms[:, None] / norms[None, :]
        np.fill_diagonal(self.sim, 0)
        self.R = R.tocsr()
        return self

    def score(self, user_idx):
        return np.asarray(self.R[user_idx].toarray()) @ self.sim


class ImplicitALS:
    """Implicit-feedback ALS (Hu, Koren, Volinsky). Small and dependency-free."""
    def __init__(self, factors=32, reg=0.1, alpha=20.0, iters=10, seed=0):
        self.f, self.reg, self.alpha, self.iters, self.seed = factors, reg, alpha, iters, seed

    def _solve(self, R, Y):
        YtY = Y.T @ Y
        out = np.zeros((R.shape[0], self.f))
        eye = self.reg * np.eye(self.f)
        for u in range(R.shape[0]):
            s, e = R.indptr[u], R.indptr[u + 1]
            if s == e:
                continue
            Yi = Y[R.indices[s:e]]
            c = self.alpha * R.data[s:e]
            A = YtY + Yi.T @ (c[:, None] * Yi) + eye
            b = Yi.T @ (1 + c)
            out[u] = np.linalg.solve(A, b)
        return out

    def fit(self, R):
        R = R.tocsr()
        rng = np.random.default_rng(self.seed)
        self.Y = rng.normal(scale=0.1, size=(R.shape[1], self.f))
        RT = R.T.tocsr()
        for _ in range(self.iters):
            self.X = self._solve(R, self.Y)
            self.Y = self._solve(RT, self.X)
        self.X = self._solve(R, self.Y)
        return self

    def score(self, user_idx):
        return self.X[user_idx] @ self.Y.T


class CFScorer:
    """Adapter: real string IDs -> matrix indices, then both CF models.
    (cf_adapter.py in the CF branch was empty, so this is the missing glue.)"""
    def fit(self, feedback):
        ui = user_item_clicks(feedback)
        self.users = pd.Index(ui["user_id"].unique())
        self.items = pd.Index(ui["itemID"].unique())
        r = self.users.get_indexer(ui["user_id"])
        c = self.items.get_indexer(ui["itemID"])
        R = sp.csr_matrix((ui["n"].to_numpy(dtype=float), (r, c)),
                          shape=(len(self.users), len(self.items)))
        self.als = ImplicitALS().fit(R)
        iicf = ItemItemCF().fit(R)
        self.ii_all = np.asarray(R @ iicf.sim, dtype=np.float32)    # users x items
        return self

    def score_pairs(self, user_ids, item_ids):
        """(als_score, item_item_score) per pair; NaN when the user or item is unknown to CF."""
        u = self.users.get_indexer(pd.Index(user_ids))
        i = self.items.get_indexer(pd.Index(item_ids))
        ok = (u >= 0) & (i >= 0)
        als = np.full(len(u), np.nan)
        ii = np.full(len(u), np.nan)
        als[ok] = np.einsum("ij,ij->i", self.als.X[u[ok]], self.als.Y[i[ok]])
        ii[ok] = self.ii_all[u[ok], i[ok]]
        return als, ii


# =============================================================================
# PART 3 - ranker (model branch)
# =============================================================================
BASE_FEATURES = ["item_sctr", "item_impr", "item_clicks", "item_age", "item_ttl", "user_fatigue"]


class ClickRanker:
    """LightGBM lambdarank, one group per event, trained on clicked events only.
    fit(train, catalog) then score(pairs). Features use only data from before each row."""

    def __init__(self, half_life_days=1.0, prior=100.0, extra_features=(), params=None,
                 n_rounds=500, early_stopping=30, valid_frac=0.15):
        self.half_life = half_life_days * DAY
        self.prior = float(prior)
        self.extra_features = list(extra_features)
        self.n_rounds = n_rounds
        self.early_stopping = early_stopping
        self.valid_frac = valid_frac
        self.params = {
            "objective": "lambdarank", "metric": "ndcg", "eval_at": [1],
            "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 50,
            "feature_fraction": 0.9, "verbose": -1,
        }
        if params:
            self.params.update(params)
        self.booster = None

    @property
    def features(self):
        return BASE_FEATURES + self.extra_features

    def fit(self, train, catalog, train_from_ts=None):
        """train_from_ts: only events at/after this time are used as training rows
        (earlier ones still feed the features). Needed when extra features such as
        CF scores are only valid for the later rows."""
        import lightgbm as lgb

        cols = ["event_id", "user_id", "timestamp", "itemID", "clicked"] + self.extra_features
        tr = train[cols].copy()
        tr["clicked"] = tr["clicked"].astype(int)
        tr = tr.drop_duplicates(["event_id", "itemID"])
        tr = tr.sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)
        self.catalog = catalog[["itemID", "valid_from", "valid_to"]].drop_duplicates("itemID")

        # state kept for scoring (everything in `train`)
        self.t_ref = float(tr["timestamp"].max())
        self.g_ctr = float(tr["clicked"].mean())
        w = np.exp2((tr["timestamp"].to_numpy(dtype=float) - self.t_ref) / self.half_life)
        self.item_state = pd.DataFrame({"itemID": tr["itemID"], "w": w, "wc": w * tr["clicked"],
                                        "n": 1, "c": tr["clicked"]}).groupby("itemID").sum()
        ev = tr.drop_duplicates("event_id").copy()
        self.user_events = ev.groupby("user_id").size()

        # causal features for the training rows: only what happened BEFORE each row
        w_s = pd.Series(w, index=tr.index)
        wc_s = w_s * tr["clicked"]
        g = tr.groupby("itemID")
        wi_b = (w_s.groupby(tr["itemID"]).cumsum() - w_s).to_numpy()
        wc_b = (wc_s.groupby(tr["itemID"]).cumsum() - wc_s).to_numpy()
        scale = self._scale(tr["timestamp"].to_numpy())
        tr["item_sctr"] = (wc_b * scale + self.prior * self.g_ctr) / (wi_b * scale + self.prior)
        tr["item_impr"] = g.cumcount().to_numpy(dtype=float)
        tr["item_clicks"] = (g["clicked"].cumsum() - tr["clicked"]).to_numpy(dtype=float)
        ev["user_fatigue"] = ev.groupby("user_id").cumcount()
        tr = tr.merge(ev[["event_id", "user_fatigue"]], on="event_id", how="left")
        tr = tr.merge(self.catalog, on="itemID", how="left")
        tr["item_age"] = tr["timestamp"] - tr["valid_from"]
        tr["item_ttl"] = tr["valid_to"] - tr["timestamp"]

        # clicked events only, grouped by event
        n_shown = tr.groupby("event_id")["clicked"].transform("size")
        any_click = tr.groupby("event_id")["clicked"].transform("max")
        keep = (any_click == 1) & (n_shown > 1)
        if train_from_ts is not None:
            keep &= tr["timestamp"] >= train_from_ts
        rows = tr[keep].reset_index(drop=True)
        if rows.empty:
            raise ValueError("no clicked events to train on")
        cut = rows.drop_duplicates("event_id")["timestamp"].quantile(1 - self.valid_frac)
        fit_rows, val_rows = rows[rows["timestamp"] < cut], rows[rows["timestamp"] >= cut]

        def dataset(df):
            sizes = df.groupby("event_id", sort=False).size().to_numpy()
            return lgb.Dataset(df[self.features], label=df["clicked"], group=sizes)

        if len(val_rows) and len(fit_rows) and self.early_stopping:
            self.booster = lgb.train(self.params, dataset(fit_rows), self.n_rounds,
                                     valid_sets=[dataset(val_rows)],
                                     callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)])
            self.best_iteration = self.booster.best_iteration
        else:
            self.booster = lgb.train(self.params, dataset(rows), 100)
            self.best_iteration = self.booster.current_iteration()
        self.n_train_events = rows["event_id"].nunique()
        return self

    def item_sctr_at(self, item_ids, t):
        """Decayed, smoothed click rate of each item as of time t."""
        st = self.item_state.reindex(pd.Index(item_ids))
        scale = self._scale(t)
        return ((st["wc"].fillna(0).to_numpy() * scale + self.prior * self.g_ctr)
                / (st["w"].fillna(0).to_numpy() * scale + self.prior))

    def make_features(self, pairs, catalog=None, request_col=None):
        p = pairs.reset_index(drop=True).copy()
        st = self.item_state.reindex(pd.Index(p["itemID"]))
        p["item_sctr"] = self.item_sctr_at(p["itemID"], p["timestamp"].to_numpy(dtype=float))
        p["item_impr"] = st["n"].fillna(0).to_numpy()
        p["item_clicks"] = st["c"].fillna(0).to_numpy()
        cat = self.catalog if catalog is None else catalog[["itemID", "valid_from", "valid_to"]]
        p = p.merge(cat.drop_duplicates("itemID"), on="itemID", how="left")
        p["item_age"] = p["timestamp"] - p["valid_from"]
        p["item_ttl"] = p["valid_to"] - p["timestamp"]
        if "user_fatigue" not in p.columns:
            p["user_fatigue"] = self.fatigue(p, request_col)
        return p[self.features]

    def score(self, pairs, catalog=None, request_col=None):
        if request_col is None:
            request_col = next((c for c in ("event_id", "rowid") if c in pairs.columns), None)
        X = self.make_features(pairs, catalog, request_col)
        return self.booster.predict(X, num_iteration=self.best_iteration)

    def fatigue(self, p, request_col=None):
        """Slates the user has seen before this request: history + earlier requests in `p`."""
        hist = p["user_id"].map(self.user_events).fillna(0).to_numpy()
        key = [request_col] if request_col else ["user_id", "timestamp"]
        req = p.drop_duplicates(key)[list(dict.fromkeys(key + ["user_id", "timestamp"]))]
        req = req.sort_values("timestamp", kind="stable")
        req["_prior"] = req.groupby("user_id").cumcount()
        prior = p[key].merge(req[key + ["_prior"]], on=key, how="left")["_prior"].to_numpy()
        return hist + prior

    def _scale(self, ts):
        return np.exp2(np.clip((self.t_ref - np.asarray(ts, dtype=float)) / self.half_life, -1000, 1000))


def fit_models(feedback, catalog, use_cf=USE_CF):
    """Train the ranker (and CF). Returns (ranker, cf); cf is None when CF is off.

    CF scores are ranker features, so the training rows need CF scores that did not
    see those rows' own clicks: CF is fitted on the oldest CF_SPLIT of events, the
    ranker trains on the rest, then CF is refitted on everything for prediction."""
    if not use_cf:
        return ClickRanker().fit(feedback, catalog), None

    t_cf = feedback.drop_duplicates("event_id")["timestamp"].quantile(CF_SPLIT)
    late = (feedback["timestamp"] >= t_cf).to_numpy()
    cf_early = CFScorer().fit(feedback[~late])
    train = feedback.copy()
    als = np.full(len(train), np.nan)
    ii = np.full(len(train), np.nan)
    als[late], ii[late] = cf_early.score_pairs(train["user_id"].to_numpy(dtype=object)[late],
                                               train["itemID"].to_numpy(dtype=object)[late])
    train["cf_als"], train["cf_item"] = als, ii
    ranker = ClickRanker(extra_features=CF_FEATURES).fit(train, catalog, train_from_ts=t_cf)
    return ranker, CFScorer().fit(feedback)


# =============================================================================
# PART 4 - predict (what the scorer calls)
# =============================================================================
def _first_valid(ts, ids, vf, vt, k):
    """For each timestamp, the first k live items in `ids` order -> (row_index, itemID)."""
    ok = (vf[None, :] <= ts[:, None]) & (ts[:, None] < vt[None, :])
    r, c = np.nonzero(ok & (ok.cumsum(axis=1) <= k))
    return r, ids[c]


def _candidates(chunk, ids, vf, vt, cf, cf_vf, cf_vt):
    ts = chunk["timestamp"].to_numpy()
    rows, items = _first_valid(ts, ids[:TOP_M], vf[:TOP_M], vt[:TOP_M], K_POP)
    if cf is not None:
        u = cf.users.get_indexer(chunk["user_id"])
        known = np.flatnonzero(u >= 0)
        if len(known):
            S = cf.als.X[u[known]] @ cf.als.Y.T
            t = ts[known][:, None]
            S[~((cf_vf[None, :] <= t) & (t < cf_vt[None, :]))] = -np.inf
            k = min(K_CF, S.shape[1])
            top = np.argpartition(-S, k - 1, axis=1)[:, :k]
            good = np.isfinite(np.take_along_axis(S, top, axis=1)).ravel()
            rows = np.concatenate([rows, np.repeat(known, k)[good]])
            items = np.concatenate([items, cf.items.to_numpy(dtype=object)[top.ravel()[good]]])
    return pd.DataFrame({"row": rows, "itemID": items}).drop_duplicates()


def predict(data_dir: str, batch_size: int = 256):
    t0 = time.time()
    requests, catalog, feedback = load_data(data_dir)
    n = len(requests)

    ranker = cf = None
    try:
        ranker, cf = fit_models(feedback, catalog)
        _log(f"ranker trained on {ranker.n_train_events} clicked events, "
             f"{ranker.best_iteration} trees, cf={'on' if cf is not None else 'off'}, "
             f"{time.time() - t0:.1f}s")
    except Exception as e:
        _log("WARNING: model training failed, using popularity only:", repr(e))

    # popularity order of the catalogue (ranker's decayed click rate, else the 24h click rate)
    try:
        if ranker is not None:
            pop = pd.Series(ranker.item_sctr_at(catalog["itemID"], float(requests["timestamp"].median())),
                            index=catalog["itemID"].to_numpy(dtype=object))
        else:
            pop = item_features(feedback).set_index("itemID")["ctr_24h"]
            pop = catalog["itemID"].map(pop).fillna(feedback["clicked"].mean())
            pop.index = catalog["itemID"].to_numpy(dtype=object)
    except Exception as e:
        _log("WARNING: popularity scores failed:", repr(e))
        pop = pd.Series(0.0, index=catalog["itemID"].to_numpy(dtype=object))
    order = np.argsort(-pop.to_numpy(), kind="stable")
    ids = catalog["itemID"].to_numpy(dtype=object)[order]
    vf, vt = catalog["valid_from"].to_numpy()[order], catalog["valid_to"].to_numpy()[order]

    cf_vf = cf_vt = None
    if ranker is not None:
        try:
            requests["user_fatigue"] = ranker.fatigue(requests, "rowid")
            if cf is not None:
                win = catalog.set_index("itemID").reindex(cf.items)
                cf_vf = win["valid_from"].fillna(np.inf).to_numpy()
                cf_vt = win["valid_to"].fillna(-np.inf).to_numpy()
        except Exception as e:
            _log("WARNING: request setup failed, using popularity only:", repr(e))
            ranker = None

    for start in range(0, n, CHUNK):
        chunk = requests.iloc[start:start + CHUNK].reset_index(drop=True)
        picks = np.full(len(chunk), None, dtype=object)

        if ranker is not None and (time.time() - t0) < TIME_BUDGET:
            try:
                pairs = _candidates(chunk, ids, vf, vt, cf, cf_vf, cf_vt)
                r = pairs["row"].to_numpy()
                pairs["user_id"] = chunk["user_id"].to_numpy(dtype=object)[r]
                pairs["timestamp"] = chunk["timestamp"].to_numpy()[r]
                pairs["user_fatigue"] = chunk["user_fatigue"].to_numpy()[r]
                if cf is not None:
                    pairs["cf_als"], pairs["cf_item"] = cf.score_pairs(
                        pairs["user_id"].to_numpy(dtype=object), pairs["itemID"].to_numpy(dtype=object))
                pairs = pairs.reset_index(drop=True)
                pairs["score"] = ranker.score(pairs, catalog)
                best = pairs.loc[pairs.groupby("row")["score"].idxmax()]
                picks[best["row"].to_numpy()] = best["itemID"].to_numpy(dtype=object)
            except Exception as e:
                _log(f"WARNING: model path failed for rows {start}+, popularity instead:", repr(e))

        # anything unanswered: most popular available item from the full catalogue
        miss = np.flatnonzero(pd.isna(picks))
        if len(miss):
            fb = pick_best_available(chunk.iloc[miss], catalog, pop)["itemID"].to_numpy(dtype=object)
            picks[miss] = fb
        # no item live at all at that timestamp: it scores 0 either way, but still answer
        picks[pd.isna(picks)] = ids[0]

        rowids = [str(x) for x in chunk["rowid"].tolist()]
        items = [str(x) for x in picks.tolist()]
        for i in range(0, len(rowids), batch_size):
            yield list(zip(rowids[i:i + batch_size], items[i:i + batch_size]))
    _log(f"done: {n} rows in {time.time() - t0:.1f}s")


# ------------------------------------------------------------------ local check
if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "round-1")
    t0 = time.time()
    out = [p for batch in predict(d) for p in batch]
    took = time.time() - t0
    req, cat, _ = load_data(d)
    pred = pd.DataFrame(out, columns=["rowid", "itemID"])
    m = pred.merge(req, on="rowid", how="left").merge(cat, on="itemID", how="left")
    live = (m["valid_from"] <= m["timestamp"]) & (m["timestamp"] < m["valid_to"])
    print(f"time: {took:.1f}s (limit 240s)")
    print(f"rows answered: {len(pred):,} / {len(req):,}")
    print("duplicate rowids:", int(pred["rowid"].duplicated().sum()), "(must be 0)")
    print("items not in catalogue:", int(m["valid_from"].isna().sum()), "(must be 0)")
    print(f"inside availability window: {live.mean():.1%}")
    print("distinct items recommended:", pred["itemID"].nunique())
"""Team Nova - Phase 2 submission (Round 1).

Model: SmoothedCTR, the best model on our validation (hit@1 0.525 vs 0.441 for
the organisers' most-clicked example). For every item we compute a recency-weighted,
smoothed click rate from the feedback in data_dir (computed at predict time, no
pre-trained weights). For every request we recommend the highest-scoring item that
is available at the request's timestamp: valid_from <= t < valid_to.
New users get the same best available item.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HALF_LIFE_DAYS = 1.0      # tuned on validation (round 1)
PRIOR_STRENGTH = 100.0    # tuned on validation (round 1)
TOP_K = 512               # candidates checked per row before falling back to a full scan
DAY = 86400


def load_feedback(data):
    """Long format: one row per displayed item, clicked 0/1."""
    pq = data / "feedback_long.parquet"
    csv = data / "feedback_long.csv"
    try:
        if pq.exists():
            return pd.read_parquet(pq, columns=["timestamp", "itemID", "clicked"])
    except Exception:
        pass
    if csv.exists():
        return pd.read_csv(csv, usecols=["timestamp", "itemID", "clicked"], dtype={"itemID": str})
    rows = []
    with open(data / "feedback.jsonl") as f:
        for line in f:
            e = json.loads(line)
            for it in e["displayed_items"]:
                rows.append((e["timestamp"], it, int(it == e.get("clicked_item"))))
    return pd.DataFrame(rows, columns=["timestamp", "itemID", "clicked"])


def item_scores(fb, items):
    fb = fb.astype({"itemID": str})
    ts = fb["timestamp"].astype("int64").to_numpy()
    w = 0.5 ** ((ts.max() - ts) / (HALF_LIFE_DAYS * DAY))
    g = pd.DataFrame({"itemID": fb["itemID"].to_numpy(), "w": w,
                      "wc": w * fb["clicked"].astype(float).to_numpy()}).groupby("itemID")[["w", "wc"]].sum()
    global_ctr = g["wc"].sum() / max(g["w"].sum(), 1e-9)
    ctr = (g["wc"] + PRIOR_STRENGTH * global_ctr) / (g["w"] + PRIOR_STRENGTH)
    return ctr.reindex(items).fillna(global_ctr).to_numpy()


def predict(data_dir, batch_size=256):
    data = Path(data_dir)
    rows = pd.read_csv(data / "x_pred.csv", dtype={"rowid": str, "user_id": str})
    rows = rows.drop_duplicates("rowid")
    cat = pd.read_csv(data / "catalog.csv", dtype={"itemID": str}).drop_duplicates("itemID")

    items = cat["itemID"].to_numpy()
    vf = cat["valid_from"].to_numpy(dtype="int64")
    vt = cat["valid_to"].to_numpy(dtype="int64")

    try:
        score = item_scores(load_feedback(data), items)
    except Exception as e:
        print("scoring failed, using catalogue order:", e)
        score = np.zeros(len(items))

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
        yield [(str(a), str(b)) for a, b in zip(r, chosen)]
