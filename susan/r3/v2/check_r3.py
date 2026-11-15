"""Round 3 submission check for v2/solution.py: runtime, row coverage, availability. Prints summary numbers only.
Run ON THE VM:  /opt/techarena/venvs/torch/bin/python check_r3.py ~/data/round-3/phase-2-round-3-input
"""
import sys, time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))      # import solution.py from this folder
import solution

D = Path(sys.argv[1])
t0 = time.time()
pairs = [p for batch in solution.predict(str(D)) for p in batch]
secs = time.time() - t0

xp = pd.read_csv(D / "x_pred.csv", dtype={"rowid": str})
cat = pd.read_csv(D / "catalog.csv", dtype={"itemID": str})
out = pd.DataFrame(pairs, columns=["rowid", "itemID"])
all_str = all(isinstance(r, str) and isinstance(i, str) for r, i in pairs)
dups = int(out.rowid.duplicated().sum())
rows_ok = len(out) == len(xp)

j = out.reset_index().merge(xp[["rowid", "timestamp"]].drop_duplicates("rowid"), on="rowid", how="left")
j = j.merge(cat[["itemID", "valid_from", "valid_to"]], on="itemID", how="left")
j["ok"] = (j.valid_from <= j.timestamp) & (j.timestamp < j.valid_to)
avail = j.groupby("index").ok.any().reindex(range(len(out)), fill_value=False).mean() if len(out) else 0.0

vc = out.itemID.value_counts()
top_share = vc.iloc[0] / len(out) if len(out) else 0.0
print(f"seconds {secs:.1f} | rows {len(out):,} vs x_pred {len(xp):,} | duplicate rowids {dups:,} | all str {all_str} "
      f"| available {avail:.2%} | distinct items {len(vc):,} | top item share {top_share:.2%}")
ok = rows_ok and dups == 0 and all_str and avail == 1.0 and secs < 240
print("R3 VERDICT: OK" if ok else "R3 VERDICT: NOT READY")
