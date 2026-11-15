import sys, importlib.util, numpy as np, pandas as pd
from pathlib import Path
R3 = "/home/user/data/round-3"; DATA = R3 + "/phase-2-round-3-input"; NUDIR = "/home/user/susan_r4/nu"
sys.path.insert(0, NUDIR)
spec = importlib.util.spec_from_file_location("nusol", NUDIR + "/solution.py"); NU = importlib.util.module_from_spec(spec); spec.loader.exec_module(NU)
X = pd.read_csv(DATA + "/x_pred.csv", dtype={"rowid": str, "user_id": str}).drop_duplicates("rowid")
L = pd.read_csv(R3 + "/phase-2-round-3-labels.csv", dtype={"event_id": str, "user_id": str, "itemID": str})
t0 = int(X.timestamp.min()); L = L[L.timestamp >= t0]
ev = L.drop_duplicates("event_id")[["event_id", "user_id", "timestamp"]].merge(X[["rowid", "user_id", "timestamp"]], on=["user_id", "timestamp"])
L = L.merge(ev[["event_id", "rowid"]], on="event_id")
slate = L.groupby("rowid").itemID.apply(set); clk = L[L.clicked == 1].drop_duplicates("rowid").set_index("rowid").itemID
def replay(name, p):
    p = pd.Series(p); inS = np.array([p.get(r) in slate[r] for r in slate.index]); c = clk[clk.index.isin(slate.index[inS])]
    print(f"{name:<46} pick in slate {inS.mean():6.1%} | clicked scored {len(c):>7,} | REPLAY {100*(p.reindex(c.index).to_numpy()==c.to_numpy()).mean():6.2f}", flush=True)
base = dict(x for b in NU.predict(DATA) for x in b); replay("Nova User (submitted for R4)", base)
fb = NU._read_feedback(Path(DATA)); cat = pd.read_csv(DATA + "/catalog.csv", dtype={"itemID": str}).drop_duplicates("itemID")
sc = NU.nu.Scorer().fit(fb, cat); ap = pd.Series(sc.ap_all, index=sc.items)
fb = fb.astype({"user_id": str, "itemID": str}); c = fb[fb.clicked.astype(int) == 1]
h = c.groupby(["user_id", "itemID"]).agg(n=("clicked", "size"), last=("timestamp", "max"), first=("timestamp", "min")).reset_index()
q = X[["rowid", "user_id", "timestamp"]].merge(h, on="user_id")
vf = q.itemID.map(cat.set_index("itemID").valid_from); vt = q.itemID.map(cat.set_index("itemID").valid_to)
q = q[(vf <= q.timestamp) & (q.timestamp < vt)].copy(); q["ap"] = ap.reindex(q.itemID).fillna(0).to_numpy()
print(f"rows with >=1 live past-clicked item: {q.rowid.nunique():,} of {len(X):,} | mean live past-clicked items per such row {len(q)/max(q.rowid.nunique(),1):.2f}")
rules = {"most RECENT click": ["last"], "most CLICKS (tie: recent)": ["n", "last"], "best APPEAL": ["ap"],
         "OLDEST click": ["first"], "most CLICKS (tie: appeal)": ["n", "ap"], "APPEAL x recency mix": ["mix"]}
q["mix"] = q.ap + 0.5 * np.log1p(q.n) - 0.1 * np.log1p((q.timestamp - q["last"]) / 3600)
for name, keys in rules.items():
    asc = [k == "first" for k in keys]
    pick = q.sort_values(keys, ascending=[not a for a in asc]).drop_duplicates("rowid").set_index("rowid").itemID if name != "OLDEST click" \
        else q.sort_values("first").drop_duplicates("rowid").set_index("rowid").itemID
    p = dict(base); p.update(pick.to_dict()); replay("always re-click: " + name, p)
