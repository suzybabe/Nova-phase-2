# Darling - round 2 submission

model.zip is what I submitted for round 2. solution.py is the same file unzipped so it's easy to read.

How it picks:
- users with a past click that's still live: Nova CF's model (cf_v3) picks among their past clicks
- users with history but nothing live (shown and never clicked, or everything expired): the item most likely to win against other teams' picks, fitted on round 1's real pooled slates
- brand new users: v2's top live item, and once itm_26ec3536 expires (~27h into the round) a stronger replacement instead of itm_7403b322

Tested with the pooled-slate replay in eval/replay on main (not hit@1). It beat v2 in every period, from +0.3 at the start to about +2.2 by day 5. v2 itself was 38.43 on the same replay.

Runs in about 11-45s, numpy/pandas/scipy/torch only.
