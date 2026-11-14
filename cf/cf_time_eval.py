import os
import numpy as np
import pandas as pd

from cf_time_weighted import TimeWeightedItemItemCF

D = os.path.expanduser("~/data/round-1/phase-2-round-1-input/")
d = pd.read_parquet(D + "feedback_long.parquet")
cat = pd.read_csv(D + "catalog.csv")

cut = d.timestamp.max() - 24 * 3600
train = d[d.timestamp < cut].copy()
test = d[d.timestamp >= cut].copy()

print("train rows", len(train), "| test rows", len(test))

items = pd.Index(sorted(set(cat.itemID) | set(d.itemID)))

prior = train.clicked.mean()
g = train.groupby("itemID").clicked.agg(["sum", "count"])
ctr = ((g["sum"] + 20 * prior) / (g["count"] + 20)).reindex(items).fillna(prior)

test["ctr"] = test.itemID.map(ctr).fillna(prior)

ev = test[test.groupby("event_id").clicked.transform("max") == 1].copy()

def hit(frame, column):
    best = frame.loc[frame.groupby("event_id")[column].idxmax()]
    return round(best.clicked.mean(), 4)

print("test events with a click:", ev.event_id.nunique())

print("CTR alone:", hit(ev, "ctr"))

for half_life in [1.0, 3.0, 7.0]:
    model = TimeWeightedItemItemCF(half_life_days=half_life)
    model.fit(train, cat)

    ev = ev.copy()
    ev["cf"] = model.score_cf(ev)

    print("\nHalf-life:", half_life, "days")
    print("CF alone:", hit(ev, "cf"))

    for w in [0.03, 0.1, 0.3, 1.0]:
        ev["combined"] = ev["ctr"] + w * ev["cf"]
        print("CTR +", w, "* CF:", hit(ev, "combined"))
