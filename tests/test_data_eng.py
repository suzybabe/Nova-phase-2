import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_eng"))

from loader import load_data, default_data_dir
from availability import pick_best_available
from features import item_features


def test_all():
    req, cat, fb = load_data(default_data_dir())
    assert req["rowid"].is_unique
    assert fb["timestamp"].dtype == "int64"

    feats = item_features(fb).set_index("itemID")
    out = pick_best_available(req, cat, feats["ctr"])
    assert len(out) == len(req)
    assert out["itemID"].notna().all()

    # every pick has to be live at its request time
    m = out.merge(req, on="rowid").merge(cat, on="itemID")
    assert ((m["valid_from"] <= m["timestamp"]) & (m["timestamp"] < m["valid_to"])).all()
    print("all good")


if __name__ == "__main__":
    test_all()