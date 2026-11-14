# pooled-slate replay

This is the script we use to score models the way the leaderboard does: every team's pick goes into one slate and you get credit for P(your item is chosen).
It's the one that matched our round 1 score (17.79 replay vs 17.49 official), so use this instead of Hit@1.

To add a model, put it in cands/<name>/solution.py (normal predict(data_dir) from the submission format) and add the name to CANDS on line 14 of v2val.py.

The aux/ folder (r1_catalog.csv, r1_old_picks.csv.gz) isn't in git because it's competition data. Get it from v2val5.zip in the team chat and drop it in here before running.

Run it on the VM:

    /opt/techarena/venvs/torch/bin/python v2val.py --data ~/r2data --cuts pre mid day5 --no-robust | tee v2val.log

---

# v2 validation bundle (Team NOVA, Tech Arena 2026 Phase 2, round 2)
Run on the VM (round 2 input unzipped at ~/r2data):

    cd ~/v2val && /opt/techarena/venvs/torch/bin/python v2val.py --data ~/r2data | tee v2val.log

Options: `--cuts pre day2 day3 mid day4 day5` (default all), `--quick N` (subsample N test rows per cut, smoke test),
`--boot 1000`, `--no-robust`, `--out DIR` (default ./results: summary.csv, segments.csv).

What it does
1. Robustness of v2 on the real R2 input: zip sha256, 75k unique rowids, items live at row time, determinism (2 runs),
   jsonl-only input, 7-row subset with unknown users, no feedback file; then every other candidate once on R2 (validity, runtime, rows != v2).
2. Forward folds on R1 (feedback.jsonl holds R1's real pooled slates): train every candidate on feedback before the cut,
   predict R1 rows after it. Cuts: pre (= all of R1), day2, day3, mid, day4, day5.
3. Pooled-slate scoring with Nova Rules' choice model (scorer_core.py = copy of r2/vm/scorer.py): keep / drop with
   paired Poisson-bootstrap CIs vs v2, judges J_pre (fit on R1 clicks before the cut, primary), J_all, J_post;
   model-free click-win share (pick == actual click).
4. Segments: single-live prior, multi-live, priors-expired, shown-never-clicked, unseen.
5. Slate sensitivity: other teams add a recent-reclick or most-clicked pick, or nobody else re-clicks.
   Note: drop already removes the global top in ~98% of rows (our R1 pick was the global top in 73,829/75,000).
Candidates: v2 (model_angle2), HIST0 (model_angle), final2 (user_angle/final2), v3, v3_gtop (referee/cand3),
cf_v3 (Nova CF, r2/cf_v3/model.zip), m3 (Nova Model, r2/model_angle3/model.zip; switches ranking path on the number of pooled slates,
so it behaves differently on early cuts), v4 (v2 + v3's pick for users with history but no live prior), v4b (v4 + v3 on multi-live rows),
v4c (cf_v3 base + v3's pick for history-no-live rows), lea_v4 (teammate's sequence-aware choice model, ~23s), cold_pooled (referee: v2 warm + v3 appeal on ALL cold rows incl. unseen, only if pooled slates exist), cold_post (referee variant, see its docstring), plus rules A recent, B72 decay-count, Lea freq-first (most-clicked live prior, ties by popularity, else most-clicked live item).
