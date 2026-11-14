import numpy as np
import pandas as pd

DAY = 86_400


class COEC:
    """Clicks Over Expected Clicks: item score adjusted for user fatigue."""

    def __init__(self, half_life_days=1.0, prior_strength=20.0, max_k=25):
        self.hl = half_life_days
        self.k = float(prior_strength)
        self.max_k = int(max_k)

    def fit(self, train, catalog):
        t = train[["event_id", "user_id", "timestamp", "itemID", "clicked"]].copy()
        ev = (t.groupby("event_id")
                .agg(user_id=("user_id", "first"), timestamp=("timestamp", "first"),
                     n=("itemID", "size"), c=("clicked", "max"))
                .sort_values(["timestamp"], kind="stable"))
        ev["slate_no"] = ev.groupby("user_id").cumcount().clip(upper=self.max_k)
        p_slate = ev.groupby("slate_no")["c"].mean()
        ev["p_item"] = ev["slate_no"].map(p_slate) / ev["n"]
        t["expected"] = t["event_id"].map(ev["p_item"]).to_numpy()
        age = (t["timestamp"].max() - t["timestamp"]) / DAY
        w = np.ones(len(t)) if self.hl is None else 0.5 ** (age / self.hl)
        g = pd.DataFrame({"i": t["itemID"], "wc": w * t["clicked"], "we": w * t["expected"]}).groupby("i")[["wc", "we"]].sum()
        self.score_map = (g["wc"] + self.k) / (g["we"] + self.k)
        self.default = 1.0
        return self

    def score(self, pairs):
        return pairs["itemID"].map(self.score_map).fillna(self.default).to_numpy(float)
