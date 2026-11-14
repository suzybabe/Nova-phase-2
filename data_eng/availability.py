import numpy as np
import pandas as pd


def available_items(catalog, t):
    # items live at time t: valid_from <= t < valid_to
    mask = (catalog["valid_from"].values <= t) & (t < catalog["valid_to"].values)
    return catalog["itemID"].values[mask]


def pick_best_available(requests, catalog, item_scores, chunk=2000):
    # for each request, pick the highest scoring item that's live at its timestamp
    # item_scores: dict or Series itemID -> score (items with no score get -inf)
    # returns a DataFrame with rowid, itemID
    scores = catalog["itemID"].map(item_scores).astype(float).fillna(-np.inf).values

    # sort items best first so argmax finds the best live one
    order = np.argsort(-scores, kind="stable")
    item_ids = catalog["itemID"].values[order]
    vf = catalog["valid_from"].values[order]
    vt = catalog["valid_to"].values[order]

    times = requests["timestamp"].values
    picks = np.empty(len(requests), dtype=object)

    for start in range(0, len(requests), chunk):
        t = times[start:start + chunk][:, None]
        live = (vf[None, :] <= t) & (t < vt[None, :])
        first = live.argmax(axis=1)
        has_any = live.any(axis=1)
        chosen = item_ids[first]
        chosen[~has_any] = None
        picks[start:start + chunk] = chosen

    return pd.DataFrame({"rowid": requests["rowid"].values, "itemID": picks})


if __name__ == "__main__":
    import time
    from loader import load_data, default_data_dir

    req, cat, fb = load_data(default_data_dir())

    print("live items at first request:", len(available_items(cat, req["timestamp"].iloc[0])))

    # quick test: plain popularity (clicks per item)
    pop = fb.groupby("itemID")["clicked"].sum()
    t0 = time.time()
    out = pick_best_available(req, cat, pop)
    print("picked for", len(out), "rows in", round(time.time() - t0, 2), "s")
    print("rows with no live item:", out["itemID"].isna().sum())
    print("unique items picked:", out["itemID"].nunique())
    print(out.head())