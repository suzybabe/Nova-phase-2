"""Validation-only wrapper for Lea's NovaRanker.
History events are outside the round's catalogue window, so for offline
validation we treat every item as available and keep scores finite."""
import numpy as np
import pandas as pd
from ranker import NovaRanker


class NovaRankerEval(NovaRanker):
    def fit(self, train, catalog):
        cat = catalog.copy()
        cat["valid_from"] = 0
        cat["valid_to"] = 10**12
        extra = set(train["itemID"].astype(str)) - set(cat["itemID"].astype(str))
        if extra:
            cat = pd.concat([cat, pd.DataFrame({"itemID": sorted(extra), "valid_from": 0, "valid_to": 10**12})])
        return super().fit(train, cat)

    def score(self, pairs):
        s = super().score(pairs)
        s[~np.isfinite(s)] = -1e9
        return s
