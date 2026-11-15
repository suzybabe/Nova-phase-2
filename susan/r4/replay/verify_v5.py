import sys, os, shutil, importlib.util, numpy as np, pandas as pd
D4 = "/home/user/data/round-4/phase-2-round-4-input"; CAT3 = "/home/user/data/round-3/phase-2-round-3-input/catalog.csv"
MODELS = {"Ady R4": "/home/user/ady_r4/ady_r4", "Nova User (was submitted)": "/home/user/susan_r4/nu", "v5 (submitted now)": "/home/user/susan_r4/v5"}
fl = pd.read_parquet(D4 + "/feedback_long.parquet").astype({"event_id": str, "user_id": str, "itemID": str})
fl["timestamp"] = fl.timestamp.astype(np.int64); fl["clicked"] = fl.clicked.astype(int)
days = sorted((fl.timestamp // 86400).unique())
def load(folder, tag):
    for m in ["solution", "scorer"]: sys.modules.pop(m, None)
    sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location("sol_" + tag, folder + "/solution.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    sys.path.pop(0); return mod
for name, cut_day, end_day in [("FOLD 1 (test 12-13 Jan)", days[-2], days[-1] + 1), ("FOLD 2 (test 11 Jan)", days[-3], days[-3] + 1)]:
    tr = fl[fl.timestamp < cut_day * 86400]; te = fl[(fl.timestamp >= cut_day * 86400) & (fl.timestamp < end_day * 86400)]
    tclk = te[te.clicked == 1].drop_duplicates("event_id").set_index("event_id").itemID
    slate = te[te.event_id.isin(tclk.index)].groupby("event_id").itemID.apply(set)
    F = os.path.expanduser("~/fold_tmp"); shutil.rmtree(F, ignore_errors=True); os.makedirs(F)
    tr.to_parquet(F + "/feedback_long.parquet", index=False); shutil.copy(CAT3, F + "/catalog.csv")
    te.drop_duplicates("event_id").loc[lambda x: x.event_id.isin(tclk.index), ["event_id", "user_id", "timestamp"]].rename(columns={"event_id": "rowid"}).to_csv(F + "/x_pred.csv", index=False)
    print(f"\n{name}: train rows {len(tr):,} | clicked test slates {len(tclk):,}", flush=True)
    P = {}
    for tag, folder in MODELS.items():
        mod = load(folder, str(abs(hash(tag))))
        P[tag] = pd.Series(dict(x for b in mod.predict(F) for x in b)).reindex(tclk.index)
        inS = np.array([p in slate[r] for r, p in P[tag].items()]); c = tclk[inS]
        print(f"  {tag:<28} pick in slate {inS.mean():6.1%} | SCORE {100*(P[tag][c.index].to_numpy()==c.to_numpy()).mean():6.2f}", flush=True)
    a, b = P["Nova User (was submitted)"], P["v5 (submitted now)"]
    both = np.array([x in slate[r] and y in slate[r] for r, x, y in zip(a.index, a, b)]); diff = both & (a.to_numpy() != b.to_numpy())
    wa = (a.to_numpy()[diff] == tclk.to_numpy()[diff]).sum(); wb = (b.to_numpy()[diff] == tclk.to_numpy()[diff]).sum()
    print(f"  PAIRED (both picks in slate, picks differ: {diff.sum():,} slates): user chose Nova User's pick {wa:,} times vs v5's pick {wb:,} times")
shutil.rmtree(os.path.expanduser("~/fold_tmp"), ignore_errors=True)
