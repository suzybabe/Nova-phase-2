import sys, importlib.util, numpy as np, pandas as pd
from pathlib import Path
R3 = "/home/user/data/round-3"; DATA = R3 + "/phase-2-round-3-input"; NUDIR = "/home/user/susan_r4/nu"
sys.path.insert(0, NUDIR)
spec = importlib.util.spec_from_file_location("nusol", NUDIR + "/solution.py"); NU = importlib.util.module_from_spec(spec); spec.loader.exec_module(NU)
X = pd.read_csv(DATA + "/x_pred.csv", dtype={"rowid": str, "user_id": str}).drop_duplicates("rowid")
L = pd.read_csv(R3 + "/phase-2-round-3-labels.csv", dtype={"event_id": str, "user_id": str, "itemID": str})
L = L[L.timestamp >= int(X.timestamp.min())]
ev = L.drop_duplicates("event_id")[["event_id", "user_id", "timestamp"]].merge(X[["rowid", "user_id", "timestamp"]], on=["user_id", "timestamp"])
L = L.merge(ev[["event_id", "rowid"]], on="event_id")
slate = L.groupby("rowid").itemID.apply(set); clk = L[L.clicked == 1].drop_duplicates("rowid").set_index("rowid").itemID
base = pd.Series(dict(x for b in NU.predict(DATA) for x in b))
fb = NU._read_feedback(Path(DATA)); cat = pd.read_csv(DATA + "/catalog.csv", dtype={"itemID": str}).drop_duplicates("itemID")
sc = NU.nu.Scorer().fit(fb, cat); ap = pd.Series(sc.ap_all, index=sc.items); pct = ap.rank(pct=True) * 100
fb = fb.astype({"user_id": str, "itemID": str}); c = fb[fb.clicked.astype(int) == 1]
h = c.groupby(["user_id", "itemID"]).agg(n=("clicked", "size"), last=("timestamp", "max")).reset_index()
xr = X.set_index("rowid")
q = X[["rowid", "user_id", "timestamp"]].merge(h, on="user_id")
vf = q.itemID.map(cat.set_index("itemID").valid_from); vt = q.itemID.map(cat.set_index("itemID").valid_to)
q = q[(vf <= q.timestamp) & (q.timestamp < vt)]
recent = q.sort_values("last", ascending=False).drop_duplicates("rowid").set_index("rowid").itemID
hh = h.set_index(["user_id", "itemID"])
cand = [r for r in clk.index if r in recent.index and base[r] != recent[r] and base[r] in slate[r] and recent[r] in slate[r] and clk[r] in (base[r], recent[r])]
rng = np.random.default_rng(7); pick = rng.choice(cand, size=min(30, len(cand)), replace=False)
print(f"decisive disagreement cases available: {len(cand):,} -> sample 30\n")
ans = []
def desc(u, i, t):
    e = f"overall appeal percentile {pct.get(i, 50):.0f}"
    if (u, i) in hh.index:
        n, last = hh.loc[(u, i)]; e += f"; this user clicked it {int(n)}x, last click {(t - last) / 3600:.0f}h before this request"
    else:
        e += "; this user never clicked it"
    return e
for k, r in enumerate(pick, 1):
    u, t = xr.at[r, "user_id"], int(xr.at[r, "timestamp"]); opts = [base[r], recent[r]]
    if rng.random() < 0.5: opts = opts[::-1]
    print(f"Case {k:>2}: Option 1 = {desc(u, opts[0], t)}\n         Option 2 = {desc(u, opts[1], t)}")
    ans.append((k, 1 + opts.index(clk[r]), 1 + opts.index(base[r])))
pd.DataFrame(ans, columns=["case", "clicked_option", "nova_user_option"]).to_csv("/home/user/judge_answers.csv", index=False)
