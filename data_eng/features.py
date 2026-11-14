import os
import numpy as np
import pandas as pd


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

    # recent windows
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


def user_features(feedback):
    events = feedback.drop_duplicates("event_id")
    clicks = feedback[feedback["clicked"] == 1]

    u = pd.DataFrame({"n_events": events.groupby("user_id").size()})
    u["n_clicks"] = clicks.groupby("user_id").size()
    u = u.fillna(0)
    u["user_ctr"] = u["n_clicks"] / u["n_events"]
    u["last_ts"] = events.groupby("user_id")["timestamp"].max()

    last_click = clicks.sort_values("timestamp").groupby("user_id")["itemID"].last()
    u["last_clicked_item"] = last_click
    return u.reset_index()


def user_item_clicks(feedback):
    # how many times each user clicked each item, for the CF people
    clicks = feedback[feedback["clicked"] == 1]
    return clicks.groupby(["user_id", "itemID"]).size().rename("n").reset_index()


def build_all(data_dir, out_dir="features_out"):
    from loader import load_data
    req, cat, fb = load_data(data_dir)

    items = item_features(fb)
    # add catalogue items that have no feedback yet, with the average ctr
    items = cat[["itemID", "valid_from", "valid_to"]].merge(items, on="itemID", how="left")
    prior = fb["clicked"].mean()
    for c in ["ctr", "ctr_24h", "ctr_6h"]:
        items[c] = items[c].fillna(prior)
    items = items.fillna(0)
    items["is_new_item"] = (items["shown"] == 0).astype(int)

    users = user_features(fb)
    ui = user_item_clicks(fb)

    os.makedirs(out_dir, exist_ok=True)
    items.to_parquet(os.path.join(out_dir, "item_features.parquet"), index=False)
    users.to_parquet(os.path.join(out_dir, "user_features.parquet"), index=False)
    ui.to_parquet(os.path.join(out_dir, "user_item_clicks.parquet"), index=False)
    return items, users, ui


if __name__ == "__main__":
    from loader import default_data_dir

    items, users, ui = build_all(default_data_dir())
    print("items:", items.shape, "new items with no feedback:", items["is_new_item"].sum())
    print("users:", users.shape)
    print("user-item click pairs:", ui.shape)
    print(items.sort_values("ctr_24h", ascending=False).head(5)[["itemID", "shown", "clicks", "ctr", "ctr_24h"]])