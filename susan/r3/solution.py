"""Team Nova - Round 3/4 solution v1: SASRec-lite starter + the organisers' recommended upgrades.

Starting point: organisers' sasrec-lite-starter (self-attention over each user's click history).
v1 changes (all switchable in CFG so the validation can compare against the starter exactly):
  1. CONTEXT-AWARE: hour-of-day and weekday of every past click (input side) and of the request
     (query side) are embedded, so the model can learn what a user wants in each context.
  2. Bigger / longer: MAXLEN 20, D 64, 2 blocks, 2 heads, cosine LR schedule, time-budgeted epochs.
  3. COLD START: users with no clicks go through the same model (start token + request context)
     instead of a random item.
  4. Robust data building: vectorised (fast), handles slates with fewer than 3 items.
Contract: predict(data_dir, batch_size) yields lists of (rowid, itemID) strings; one row each;
items from catalog.csv and available at the row timestamp. Trains inside predict, < 240 s.
"""
import math, time, sys
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn

CFG = dict(MAXLEN=20, D=64, HEADS=2, BLOCKS=2, DROP=0.1, EPOCHS=12, LR=2e-3, BATCH=2048,
           TIME_BUDGET_S=130.0, CTX=True, COLD="model", SEED=0)
STARTER_CFG = dict(MAXLEN=10, D=32, HEADS=1, BLOCKS=1, DROP=0.1, EPOCHS=8, LR=2e-3, BATCH=2048,
                   TIME_BUDGET_S=150.0, CTX=False, COLD="random", SEED=0, SCHEDULE=False)


def log(*a):
    print("[nova-r3]", *a, file=sys.stderr, flush=True)


def ctx_of(ts):
    ts = np.asarray(ts, dtype=np.int64)
    return ((ts // 3600) % 24).astype(np.int64), (((ts // 86400) + 3) % 7).astype(np.int64)   # hour, weekday (Mon=0)


def load(data_dir):
    data = Path(data_dir)
    fl = pd.read_parquet(data / "feedback_long.parquet") if (data / "feedback_long.parquet").exists() \
        else pd.read_csv(data / "feedback_long.csv")
    fl = fl[["event_id", "user_id", "timestamp", "itemID", "clicked"]].copy()
    fl["event_id"] = fl["event_id"].astype(str); fl["user_id"] = fl["user_id"].astype(str)
    fl["itemID"] = fl["itemID"].astype(str); fl["timestamp"] = fl["timestamp"].astype(np.int64)
    fl["clicked"] = fl["clicked"].astype(np.int64)
    cat = pd.read_csv(data / "catalog.csv", dtype={"itemID": str})
    xp = pd.read_csv(data / "x_pred.csv", dtype={"rowid": str, "user_id": str})
    xp["timestamp"] = xp["timestamp"].astype(np.int64)
    return fl, cat, xp


def build_history(clk_user, clk_ts, clk_item, q_user, q_ts, L):
    """For each query (user, ts): [START] + up to L-1 most recent clicks strictly before ts (oldest->newest).
    clk_* must be sorted by (user, ts). Returns seq item indices (Q,L), click ts (Q,L), length n (Q,)."""
    BIG = np.int64(4_000_000_000)
    ck = clk_user.astype(np.int64) * BIG + clk_ts.astype(np.int64)
    qk = q_user.astype(np.int64) * BIG + q_ts.astype(np.int64)
    k = np.searchsorted(ck, qk, side="left")
    start = np.searchsorted(ck, q_user.astype(np.int64) * BIG, side="left")
    n = np.minimum(k - start, L - 1)
    Q = len(q_user)
    seq = np.zeros((Q, L), np.int64); sts = np.zeros((Q, L), np.int64)
    for p in range(1, L):
        valid = p <= n
        idx = np.where(valid, k - n + (p - 1), 0)
        if len(clk_item):
            seq[:, p] = np.where(valid, clk_item[np.clip(idx, 0, len(clk_item) - 1)], 0)
            sts[:, p] = np.where(valid, clk_ts[np.clip(idx, 0, len(clk_ts) - 1)], 0)
    return seq, sts, n + 1          # +1 for the START token at position 0


class NovaSeq:
    def __init__(self, cfg=None):
        self.cfg = dict(CFG if cfg is None else cfg)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------ fit
    def fit(self, fl, cat, t0=None):
        c = self.cfg; t0 = time.time() if t0 is None else t0
        torch.manual_seed(c["SEED"]); np.random.seed(c["SEED"])
        items = sorted(set(fl.itemID) | set(cat.itemID))
        self.iv = {it: i + 1 for i, it in enumerate(items)}
        V = len(items); self.V = V; self.START = V + 1
        users = pd.Index(sorted(set(fl.user_id)))
        self.users = users
        clk = fl[fl.clicked == 1].sort_values(["user_id", "timestamp"], kind="stable")
        self.clk_u = users.get_indexer(clk.user_id).astype(np.int64)
        self.clk_t = clk.timestamp.to_numpy(np.int64)
        self.clk_i = clk.itemID.map(self.iv).to_numpy(np.int64)
        self.has_clicks = set(clk.user_id)
        # events -> (E, K) item matrix with mask
        f = fl.sort_values(["event_id", "itemID"], kind="stable")
        e, ev_ids = pd.factorize(f.event_id)
        pos = f.groupby(e).cumcount().to_numpy()
        K = int(pos.max()) + 1; E = int(e.max()) + 1
        ITM = np.zeros((E, K), np.int64); LAB = np.zeros((E, K), np.float32); MSK = np.zeros((E, K), bool)
        ITM[e, pos] = f.itemID.map(self.iv).to_numpy(); LAB[e, pos] = f.clicked.to_numpy(np.float32); MSK[e, pos] = True
        first = f.groupby(e).head(1)
        ev_u = users.get_indexer(first.user_id).astype(np.int64); ev_t = first.timestamp.to_numpy(np.int64)
        L = c["MAXLEN"]
        SEQ, STS, LN = build_history(self.clk_u, self.clk_t, self.clk_i, ev_u, ev_t, L)
        SEQ[:, 0] = self.START
        log(f"{E} events, {V} items, K={K}, build {time.time()-t0:.1f}s")
        self.model = self._make_model().to(self.dev)
        self._train(SEQ, STS, LN, ev_t, ITM, LAB, MSK, t0)
        return self

    def _make_model(self):
        c = self.cfg; V, D, L = self.V, c["D"], c["MAXLEN"]

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.item = nn.Embedding(V + 2, D, padding_idx=0); s.bias = nn.Embedding(V + 2, 1, padding_idx=0)
                s.pos = nn.Embedding(L, D); s.drop = nn.Dropout(c["DROP"])
                s.h_in = nn.Embedding(24, D); s.w_in = nn.Embedding(7, D)
                s.h_q = nn.Embedding(24, D); s.w_q = nn.Embedding(7, D)
                lyr = nn.TransformerEncoderLayer(D, c["HEADS"], 2 * D, c["DROP"], activation="gelu",
                                                 batch_first=True, norm_first=True)
                s.enc = nn.TransformerEncoder(lyr, c["BLOCKS"]); s.ln = nn.LayerNorm(D)
                for emb in (s.item, s.pos, s.h_in, s.w_in, s.h_q, s.w_q):
                    nn.init.normal_(emb.weight, std=0.02)
                nn.init.zeros_(s.bias.weight)
                with torch.no_grad(): s.item.weight[0].zero_()

            def h(s, seq, sts_h, sts_w, ln, qh, qw):
                B, Lq = seq.shape
                x = s.item(seq) * math.sqrt(D) + s.pos(torch.arange(Lq, device=seq.device).expand(B, Lq))
                if c["CTX"]:
                    x = x + (s.h_in(sts_h) + s.w_in(sts_w)) * (seq > 0).unsqueeze(-1)
                x = s.drop(x)
                m = torch.triu(torch.ones(Lq, Lq, device=seq.device, dtype=torch.bool), 1)
                x = s.ln(s.enc(x, mask=m, src_key_padding_mask=(seq == 0)))
                out = x[torch.arange(B, device=seq.device), ln - 1, :]
                if c["CTX"]:
                    out = out + s.h_q(qh) + s.w_q(qw)
                return out

            def score(s, hv, it):
                return (hv[:, None, :] * s.item(it)).sum(-1) + s.bias(it).squeeze(-1)
        return Net()

    def _tensors(self, SEQ, STS, LN, QT, b):
        dev = self.dev
        sh, sw = ctx_of(STS[b]); qh, qw = ctx_of(QT[b])
        T = lambda a: torch.as_tensor(a, device=dev)
        return T(SEQ[b]), T(sh), T(sw), T(LN[b]), T(qh), T(qw)

    def _train(self, SEQ, STS, LN, QT, ITM, LAB, MSK, t0):
        c = self.cfg; m = self.model; dev = self.dev
        opt = torch.optim.AdamW(m.parameters(), lr=c["LR"], weight_decay=1e-5)
        E = len(SEQ); steps_ep = math.ceil(E / c["BATCH"]); total = steps_ep * c["EPOCHS"]
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["LR"], total_steps=total, pct_start=0.1) \
            if c.get("SCHEDULE", True) else None
        npos = max(float(LAB[MSK].sum()), 1.0); pw = torch.tensor([(MSK.sum() - npos) / npos], device=dev)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        idx = np.arange(E); step = 0
        for ep in range(1, c["EPOCHS"] + 1):
            if time.time() - t0 > c["TIME_BUDGET_S"]: break
            m.train(); np.random.shuffle(idx)
            for s in range(0, E, c["BATCH"]):
                if time.time() - t0 > c["TIME_BUDGET_S"]: break
                b = idx[s:s + c["BATCH"]]
                hv = m.h(*self._tensors(SEQ, STS, LN, QT, b))
                lg = m.score(hv, torch.as_tensor(ITM[b], device=dev))
                mk = torch.as_tensor(MSK[b], device=dev)
                l = (lossf(lg, torch.as_tensor(LAB[b], device=dev)) * mk).sum() / mk.sum()
                opt.zero_grad(); l.backward(); opt.step(); step += 1
                if sched is not None and step < total: sched.step()
            log(f"epoch {ep} loss {float(l):.4f} ({time.time()-t0:.1f}s)")
        m.eval()

    # ------------------------------------------------------------------ scoring
    def _query(self, users, ts):
        u = self.users.get_indexer(pd.Index(users)).astype(np.int64)
        known = u >= 0
        u2 = np.where(known, u, -7)                       # unknown users get no history
        SEQ, STS, LN = build_history(self.clk_u, self.clk_t, self.clk_i, u2, np.asarray(ts, np.int64), self.cfg["MAXLEN"])
        SEQ[:, 0] = self.START
        return SEQ, STS, LN

    def score_items(self, users, ts, items, chunk=4096):
        """users (Q,), ts (Q,), items (Q,K) itemID strings -> scores (Q,K)."""
        it = np.vectorize(lambda x: self.iv.get(x, 0))(np.asarray(items, dtype=object)).astype(np.int64)
        SEQ, STS, LN = self._query(users, ts); QT = np.asarray(ts, np.int64)
        out = np.zeros(it.shape, np.float32)
        with torch.no_grad():
            for s in range(0, len(QT), chunk):
                b = np.arange(s, min(s + chunk, len(QT)))
                hv = self.model.h(*self._tensors(SEQ, STS, LN, QT, b))
                out[b] = self.model.score(hv, torch.as_tensor(it[b], device=self.dev)).cpu().numpy()
        if self.cfg["COLD"] == "random":
            cold = ~np.isin(np.asarray(users), list(self.has_clicks))
            out[cold] = np.random.default_rng(self.cfg["SEED"]).random((cold.sum(), it.shape[1]))
        return out

    def catalog_scores(self, users, ts, cat, chunk=1024):
        """Yields (row slice, scores (B, n_catalog)) with unavailable items set to -inf."""
        cat_items = cat.itemID.to_numpy(); vf = cat.valid_from.to_numpy(np.int64); vt = cat.valid_to.to_numpy(np.int64)
        cvi = torch.as_tensor(np.array([self.iv.get(x, 0) for x in cat_items], np.int64), device=self.dev)
        with torch.no_grad():
            cemb = self.model.item(cvi); cb = self.model.bias(cvi).squeeze(-1)
        SEQ, STS, LN = self._query(users, ts); QT = np.asarray(ts, np.int64)
        users = np.asarray(users); coldmask = ~np.isin(users, list(self.has_clicks))
        rng = np.random.default_rng(self.cfg["SEED"])
        for s in range(0, len(QT), chunk):
            b = np.arange(s, min(s + chunk, len(QT)))
            with torch.no_grad():
                hv = self.model.h(*self._tensors(SEQ, STS, LN, QT, b))
                sc = (hv @ cemb.t() + cb).cpu().numpy()
            if self.cfg["COLD"] == "random" and coldmask[b].any():
                sc[coldmask[b]] = rng.random((coldmask[b].sum(), sc.shape[1]))
            t = QT[b][:, None]
            sc[~((vf[None, :] <= t) & (t < vt[None, :]))] = -np.inf
            yield b, sc

    def recommend(self, xp, cat):
        cat_items = cat.itemID.to_numpy(); recs = np.empty(len(xp), dtype=object)
        for b, sc in self.catalog_scores(xp.user_id.to_numpy(), xp.timestamp.to_numpy(), cat):
            best = sc.argmax(1)
            recs[b] = cat_items[best]
            dead = ~np.isfinite(sc[np.arange(len(b)), best])      # nothing available: any live item / first item
            for j in np.flatnonzero(dead):
                t = xp.timestamp.iloc[b[j]]
                live = np.flatnonzero((cat.valid_from.to_numpy() <= t) & (t < cat.valid_to.to_numpy()))
                recs[b[j]] = cat_items[live[0] if len(live) else 0]
        return recs


def predict(data_dir, batch_size=256):
    t0 = time.time()
    fl, cat, xp = load(data_dir)
    xp = xp.drop_duplicates("rowid").reset_index(drop=True)
    model = NovaSeq(CFG).fit(fl, cat, t0)
    recs = model.recommend(xp, cat)
    log(f"done {time.time()-t0:.1f}s on {model.dev}")
    out = []
    for rid, it in zip(xp.rowid.to_numpy(), recs):
        out.append((str(rid), str(it)))
        if len(out) >= batch_size:
            yield out; out = []
    if out:
        yield out


if __name__ == "__main__":
    n = sum(len(b) for b in predict(sys.argv[1] if len(sys.argv) > 1 else "."))
    print("rows:", n)
