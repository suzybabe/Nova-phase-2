"""Plain LightGBM baseline on the raw round 1 data.
No cleaning. The model uses ONE feature: the smoothed click rate per item (smooth_ctr).

Usage:  python train_smoothctr.py              (expects data/round-1 next to this file)
        python train_smoothctr.py path/to/data
"""
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).parent
data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data" / "round-1"

# ---- load the raw data: one row per displayed item, clicked = 0/1 -----------
df = pd.read_csv(data_dir / "feedback_long.csv")
print(df.head())
print("rows, columns:", df.shape)
print(df["clicked"].value_counts())

# ---- split by time: oldest 80% to train, newest 20% to test ------------------
df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
cut = df["timestamp"].quantile(0.8)
train = df[df["timestamp"] < cut].copy()
test = df[df["timestamp"] >= cut].copy()
print("train rows:", len(train), " test rows:", len(test))

# ---- smoothed CTR per item ----------------------------------------------------
# smooth_ctr = (clicks + PRIOR * average_ctr) / (times_shown + PRIOR)
# An item shown only a few times is pulled towards the average click rate.
PRIOR = 20
average_ctr = train["clicked"].mean()

# Train rows: use only what happened BEFORE each row (running totals),
# otherwise the feature would contain the answer it is trying to predict.
g = train.groupby("itemID")["clicked"]
clicks_before = g.cumsum() - train["clicked"]
shown_before = g.cumcount()
train["smooth_ctr"] = (clicks_before + PRIOR * average_ctr) / (shown_before + PRIOR)

# Test rows: totals from the whole training period (what we would know at prediction time).
totals = train.groupby("itemID")["clicked"].agg(["sum", "count"])
item_ctr = (totals["sum"] + PRIOR * average_ctr) / (totals["count"] + PRIOR)
test["smooth_ctr"] = test["itemID"].map(item_ctr).fillna(average_ctr)

features = ["smooth_ctr"]   # the only feature the model sees

d_train = lgb.Dataset(train[features], label=train["clicked"])
d_test = lgb.Dataset(test[features], label=test["clicked"])

# ---- train -------------------------------------------------------------------
lgbm_params = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "verbose": -1,
}

start = datetime.now()
clf = lgb.train(lgbm_params, d_train, num_boost_round=500, valid_sets=[d_test],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(25)])
print("Training time:", datetime.now() - start)

# ---- predict on the test period ----------------------------------------------
test["score"] = clf.predict(test[features], num_iteration=clf.best_iteration)
print("AUC:", round(roc_auc_score(test["clicked"], test["score"]), 4))

# The competition question: of the items shown, do we pick the one that got clicked?
clicked_events = test[test.groupby("event_id")["clicked"].transform("max") == 1]


def top1_hit(col):
    best = clicked_events.loc[clicked_events.groupby("event_id")[col].idxmax()]
    return round(best["clicked"].mean(), 4)


print("Top-1 hit rate, LightGBM:        ", top1_hit("score"))
print("Top-1 hit rate, smooth_ctr alone:", top1_hit("smooth_ctr"))
print("Top-1 hit rate, random guess:    ", round(1 / clicked_events.groupby("event_id").size().mean(), 4))

# ---- save the model and the test predictions ---------------------------------
clf.save_model(str(HERE / "lgbm_smoothctr.txt"), num_iteration=clf.best_iteration)
test[["event_id", "user_id", "timestamp", "itemID", "clicked", "smooth_ctr", "score"]].to_csv(
    HERE / "valid_predictions_smoothctr.csv", index=False)
print("Saved lgbm_smoothctr.txt and valid_predictions_smoothctr.csv in", HERE)
