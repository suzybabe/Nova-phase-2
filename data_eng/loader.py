import os
import sys
import json
import pandas as pd


def read_table(data_dir, name):
    # parquet is faster, fall back to csv if it's missing
    pq = os.path.join(data_dir, name + ".parquet")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    return pd.read_csv(os.path.join(data_dir, name + ".csv"))


def load_data(data_dir):
    requests = read_table(data_dir, "x_pred")
    catalog = read_table(data_dir, "catalog")
    feedback = read_table(data_dir, "feedback_long")

    requests["timestamp"] = requests["timestamp"].astype("int64")
    catalog["valid_from"] = catalog["valid_from"].astype("int64")
    catalog["valid_to"] = catalog["valid_to"].astype("int64")
    feedback["timestamp"] = feedback["timestamp"].astype("int64")
    feedback["clicked"] = feedback["clicked"].astype("int8")

    # feedback is cumulative, so drop any repeated rows
    feedback = feedback.drop_duplicates(subset=["event_id", "itemID"])

    return requests, catalog, feedback


def load_config(data_dir):
    with open(os.path.join(data_dir, "round_config.json")) as f:
        return json.load(f)


def default_data_dir():
    # VM path first, then my Mac's Downloads copy
    vm = os.path.expanduser("~/data/round-1/input")
    mac = os.path.expanduser("~/Downloads/phase-2-round-1-input")
    if os.path.exists(vm):
        return vm
    return mac


if __name__ == "__main__":
    if len(sys.argv) > 1:
        d = sys.argv[1]
    else:
        d = default_data_dir()

    print("loading from:", d)
    req, cat, fb = load_data(d)
    print("requests:", req.shape)
    print("catalog:", cat.shape)
    print("feedback:", fb.shape)
    print("events:", fb["event_id"].nunique(), "clicks:", fb["clicked"].sum())
    new_users = ~req["user_id"].isin(fb["user_id"])
    print("cold start users in x_pred:", round(new_users.mean() * 100, 1), "%")