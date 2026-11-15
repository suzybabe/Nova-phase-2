"""Round 3 data sanity check: prints ONLY summary counts / percentages (no raw rows, no IDs).
Run ON THE VM:  /opt/techarena/venvs/torch/bin/python data_check.py ~/data/round-3/phase-2-round-3-input
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

D = Path(sys.argv[1])
problems = []


def flag(name, n, expected=0):
    """Record a problem when a count differs from what we expect."""
    if n != expected:
        problems.append(f"{name}: {n:,} (expected {expected:,})")


def pct(a, b):
    return f"{100.0 * a / b:.2f}%" if b else "n/a"


def day(ts):
    return pd.to_datetime(np.asarray(ts, dtype=np.int64), unit="s").strftime("%Y-%m-%d %a")


# ---------------------------------------------------------------- load
fl = pd.read_parquet(D / "feedback_long.parquet") if (D / "feedback_long.parquet").exists() \
    else pd.read_csv(D / "feedback_long.csv")
cat = pd.read_csv(D / "catalog.csv", dtype={"itemID": str})
xp = pd.read_csv(D / "x_pred.csv", dtype={"rowid": str, "user_id": str})
cfg = json.loads((D / "round_config.json").read_text())
for c in ("event_id", "user_id", "itemID"):
    fl[c] = fl[c].astype(str)

# ---------------------------------------------------------------- catalog
print("=== CATALOG")
n_cat = len(cat)
dup_cat = int(cat.itemID.duplicated().sum())
bad_win = int((cat.valid_from >= cat.valid_to).sum())
print(f"rows {n_cat:,} | duplicate itemID {dup_cat:,} | valid_from >= valid_to {bad_win:,}")
flag("catalog duplicate itemID", dup_cat); flag("catalog valid_from >= valid_to", bad_win)
c1 = cat.drop_duplicates("itemID").set_index("itemID")

# ---------------------------------------------------------------- feedback
print("\n=== FEEDBACK")
n = len(fl)
print(f"rows {n:,} | events {fl.event_id.nunique():,} | users {fl.user_id.nunique():,} | items {fl.itemID.nunique():,} "
      f"| click rate {fl.clicked.mean():.4f}")
nulls = fl.isna().sum()
print("nulls per column:", ", ".join(f"{k} {v:,}" for k, v in nulls.items()))
flag("feedback nulls", int(nulls.sum()))
dup = int(fl.duplicated(["event_id", "itemID"]).sum())
print(f"duplicate (event_id, itemID) rows {dup:,} ({pct(dup, n)})")
flag("feedback duplicate (event_id, itemID)", dup)

g = fl.groupby("event_id")
size = g.size()
print("slate size distribution:", ", ".join(f"{k} items: {v:,} ({pct(v, len(size))})"
                                            for k, v in size.value_counts().sort_index().items()))
flag("slates not of size 3", int((size != 3).sum()))
nclk = g.clicked.sum()
c0, c1n, c2 = int((nclk == 0).sum()), int((nclk == 1).sum()), int((nclk >= 2).sum())
print(f"events with 0 clicks {c0:,} ({pct(c0, len(nclk))}) | 1 click {c1n:,} ({pct(c1n, len(nclk))}) "
      f"| 2+ clicks {c2:,} ({pct(c2, len(nclk))})")
flag("events with 2+ clicks", c2)
mu = int((g.user_id.nunique() > 1).sum()); mt = int((g.timestamp.nunique() > 1).sum())
print(f"events with >1 user {mu:,} | events with >1 timestamp {mt:,}")
flag("events with >1 user", mu); flag("events with >1 timestamp", mt)

ts = fl.timestamp.astype(np.int64)
print(f"timestamp range {pd.to_datetime(ts.min(), unit='s')} -> {pd.to_datetime(ts.max(), unit='s')} (UTC)")
first = fl.drop_duplicates("event_id")
per_day = first.groupby(day(first.timestamp.astype(np.int64))).agg(events=("event_id", "size"))
clk_day = fl.groupby(day(ts)).clicked.mean()
for d, r in per_day.iterrows():
    print(f"  {d}: events {r.events:,} | click rate {clk_day[d]:.4f}")

notcat = ~fl.itemID.isin(c1.index)
print(f"impressions of items not in catalog {int(notcat.sum()):,} ({pct(notcat.sum(), n)})")
flag("impressions of items not in catalog", int(notcat.sum()))
vf = fl.itemID.map(c1.valid_from); vt = fl.itemID.map(c1.valid_to)
outside = int((~notcat & ~((vf <= ts) & (ts < vt))).sum())
print(f"impressions outside item's valid window {outside:,} ({pct(outside, n - notcat.sum())})")
flag("impressions outside valid window", outside)

ev_u = first.user_id.value_counts()
k = max(1, int(np.ceil(0.01 * len(ev_u))))
top = ev_u.index[:k]
is_top = fl.user_id.isin(top)
print(f"top-1% users ({k:,}): {pct(ev_u.iloc[:k].sum(), ev_u.sum())} of events | click rate "
      f"{fl.clicked[is_top].mean():.4f} vs others {fl.clicked[~is_top].mean():.4f}")

# ---------------------------------------------------------------- x_pred
print("\n=== X_PRED")
xn = len(xp)
xdup = int(xp.rowid.duplicated().sum()); xnull = int(xp.isna().sum().sum())
print(f"rows {xn:,} | users {xp.user_id.nunique():,} | duplicate rowid {xdup:,} | nulls {xnull:,}")
flag("x_pred duplicate rowid", xdup); flag("x_pred nulls", xnull)
xt = xp.timestamp.astype(np.int64)
print(f"timestamp range {pd.to_datetime(xt.min(), unit='s')} -> {pd.to_datetime(xt.max(), unit='s')} (UTC)")
for d, v in xt.groupby(day(xt)).size().items():
    print(f"  {d}: rows {v:,}")


def find_window(obj, path=""):
    """Look through round_config for start/end values (epoch seconds or date strings)."""
    found = {}
    if isinstance(obj, dict):
        for kk, vv in obj.items():
            p = f"{path}.{kk}" if path else kk
            if isinstance(vv, (dict, list)):
                found.update(find_window(vv, p))
            elif set(kk.lower().replace("-", "_").split("_")) & {"start", "begin", "from"}:
                found.setdefault("start", (p, vv))
            elif set(kk.lower().replace("-", "_").split("_")) & {"end", "until", "to"}:
                found.setdefault("end", (p, vv))
    return found


def to_epoch(v, is_end):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return int(v)
    t = pd.Timestamp(str(v))
    t = t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t
    if is_end and len(str(v)) <= 10:          # date only -> whole day included
        t = t + pd.Timedelta(days=1)
        return int(t.timestamp()) - 1
    return int(t.timestamp())


print("round_config keys:", sorted(cfg.keys()) if isinstance(cfg, dict) else type(cfg).__name__)
win = find_window(cfg)
if "start" in win and "end" in win:
    try:
        s, e = to_epoch(win["start"][1], False), to_epoch(win["end"][1], True)
        out_w = int(((xt < s) | (xt > e)).sum())
        print(f"window from config ({win['start'][0]} .. {win['end'][0]}): {pd.to_datetime(s, unit='s')} -> "
              f"{pd.to_datetime(e, unit='s')} | rows outside {out_w:,}")
        flag("x_pred rows outside round_config window", out_w)
    except Exception as ex:
        print("could not read window from round_config:", type(ex).__name__)
        problems.append("round_config window not readable - check by hand")
else:
    print("no start/end keys found in round_config")
    problems.append("round_config window not found - check by hand")

cold = ~xp.user_id.isin(set(fl.user_id))
print(f"x_pred users not in feedback {pct(xp.user_id[cold].nunique(), xp.user_id.nunique())} of users "
      f"({pct(cold.sum(), xn)} of rows)")
ok = cat.valid_from < cat.valid_to
svf, svt = np.sort(cat.valid_from[ok].to_numpy(np.int64)), np.sort(cat.valid_to[ok].to_numpy(np.int64))
avail = np.searchsorted(svf, xt, "right") - np.searchsorted(svt, xt, "right")   # #items with vf <= t < vt
zero = int((avail <= 0).sum())
print(f"rows with ZERO available catalogue items {zero:,} | available items per row: "
      f"min {avail.min():,} median {int(np.median(avail)):,} max {avail.max():,}")
flag("x_pred rows with zero available items", zero)

print("\nPROBLEMS:")
print("\n".join(f"- {p}" for p in problems) if problems else "- none")
