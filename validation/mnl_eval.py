import numpy as np
import pandas as pd

import nova_user_solution as sol


class MNLBlend:
    def __init__(self, w_mnl=0.7):
        self.w = float(w_mnl)

    def fit(self, train, catalog):
        fb = train.astype({"itemID": str, "user_id": str, "event_id": str})
        items = pd.unique(pd.concat([fb["itemID"], catalog["itemID"].astype(str)]))
        ctr = sol.smoothed_ctr(fb, items)
        b = sol.fit_mnl(fb).reindex(items).fillna(0.0).to_numpy()
        self.s = pd.Series(self.w * b + (1 - self.w) * np.log(ctr), index=items)
        self.default = float(self.s.min())
        return self

    def score(self, pairs):
        return pairs["itemID"].astype(str).map(self.s).fillna(self.default).to_numpy(float)
