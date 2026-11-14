import pandas as pd


def time_split(feedback, holdout_hours=24):
    # train on everything before the cutoff, test on the last X hours
    cutoff = feedback["timestamp"].max() - holdout_hours * 3600
    train = feedback[feedback["timestamp"] < cutoff]
    test = feedback[feedback["timestamp"] >= cutoff]
    return train, test


def slate_hit_rate(test, picks):
    # rough offline check: on clicked events, was our pick the item that got clicked?
    # picks: dict user_id -> itemID
    clicked = test[test["clicked"] == 1]
    guess = clicked["user_id"].map(picks)
    return (guess == clicked["itemID"]).mean()


def slate_choice_score(test, item_scores):
    # closer to the real metric: on clicked slates, share of the slate's score
    # that our scores put on the item that was actually clicked
    events = test[test["event_id"].isin(test.loc[test["clicked"] == 1, "event_id"])].copy()
    events["s"] = events["itemID"].map(item_scores).fillna(0).clip(lower=1e-6)
    events["p"] = events["s"] / events.groupby("event_id")["s"].transform("sum")
    return events.loc[events["clicked"] == 1, "p"].mean()


if __name__ == "__main__":
    from loader import load_data, default_data_dir
    from features import item_features

    req, cat, fb = load_data(default_data_dir())
    train, test = time_split(fb, 24)
    print("train rows:", len(train), "test rows:", len(test))

    feats = item_features(train).set_index("itemID")
    for col in ["ctr", "ctr_24h", "ctr_6h"]:
        score = slate_choice_score(test, feats[col])
        print(col, "choice score on holdout:", round(score, 4))
    print("random guess would be about:", round(1 / 3, 4))