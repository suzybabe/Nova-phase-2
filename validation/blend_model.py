"""Blend of SmoothedCTR with a second item score, both on a log scale.
score = (1 - w) * log(SmoothedCTR) + w * other
other = "choice": ChoiceLogit appeal; other = "coec": log(COEC)."""
import numpy as np

from validation import SmoothedCTR
from choice_model import ChoiceLogit
from coec_model import COEC


class Blend:
    def __init__(self, w=0.3, other="choice", half_life_days=1.0, prior_strength=100.0,
                 reg_item=1.0, coec_prior=5.0):
        self.w = float(w)
        self.other = other
        self.ctr = SmoothedCTR(half_life_days=half_life_days, prior_strength=prior_strength)
        if other == "coec":
            self.b = COEC(half_life_days=half_life_days, prior_strength=coec_prior)
        else:
            self.b = ChoiceLogit(dim=0, reg_item=reg_item, half_life_days=half_life_days)

    def fit(self, train, catalog):
        self.ctr.fit(train, catalog)
        self.b.fit(train, catalog)
        return self

    def score(self, pairs):
        a = np.log(np.clip(np.asarray(self.ctr.score(pairs), float), 1e-9, None))
        b = np.asarray(self.b.score(pairs), float)
        if self.other == "coec":
            b = np.log(np.clip(b, 1e-9, None))
        return (1 - self.w) * a + self.w * b
