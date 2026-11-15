# Round 3 - improvements and ideas on Susan's v2 (Ady)

Data: round 3 feedback (2M events, 20k users, random 3-item slates), validation = two chronological folds
(train < 7 Jan test 7 Jan; train < 8 Jan test 8 Jan; 15.7k / 13.6k clicked slates). 95% margin on hit@1 is about +/-0.005-0.008,
so differences below ~0.5 points are noise. Reference: v2 = 0.4232 / 0.4255, CTR only = 0.4216 / 0.4161, random 0.333.

## Data checks (nothing to clean)
- No nulls, no duplicate (event, item), 100 events per user, 18.6k slates with <3 items (availability).
- Click rate per event halves every ~3h at the start (53% -> 2.1%); 73% of clicks are on 6 Jan. Irrelevant to the score
  (P(chosen | click)) but it means impression-based CTR is dominated by day 1.
- 545 catalogue items were never shown in the feedback; they only get the prior CTR (fine, they rank low).

## Tested - no reliable gain
| Idea | Result |
|---|---|
| Click-conditional win-rate (ignores base-rate drift) instead of CTR | 0.412-0.415 vs 0.417: no better |
| Re-tune prior / half-life (25 combos x 2 folds) | v2 (200, 1.0, 0.5) is already the best average (0.4244) |
| Hour-of-day item CTR | 0.396: worse (too sparse) |
| Item-to-item transitions (last click, last 3) | cover only 0.4% / 1.3% of slates, no gain |
| Shrink co-click by user history length | -0.001 to +0.000: nothing |
| Penalise items already shown to the user in the logger data | +/-0.0004: nothing |
| Explicit re-click boost | no change (re-clicked items already rank first when present) |

## Tested - small, not significant (in solution_v3_experimental.py, SVD_W)
- Rank-32 SVD taste signal on the user-item click matrix, weight 0.25: +0.011 on 8 Jan, -0.001 on 7 Jan (mean +0.005).
  Retest when more days of data exist.

## Repetition / fatigue (REPEAT_PEN in solution_v3_experimental.py)
- Round 1 pooled slates: P(top item is the clicked one | click) falls with how often the user has already seen it:
  16.5% (never) -> 10% (1-2) -> 6% (3-5) -> 4% (6-10) -> 2.7% (11-20). Strong repeat decay. Caveat: slates also grew over
  those days, so part of the fall is just bigger slates.
- v2 gives each user ~1.04 distinct items over their 20 requests (mean 9.4 earlier repeats per rec).
- Penalty REPEAT_PEN*log1p(times already recommended to this user): at 0.3 it costs ~0 item quality (mean CTR of picks unchanged
  at 0.0137) and cuts repeats a lot. Round 3 data cannot confirm the decay (random slates, almost no repeats), and R3 re-clicks are
  rare (0.3%), so the simulator may behave differently. Treat as a hedge.

## Not tested yet (ideas)
- Use a past round to calibrate the repeat penalty: fit the seen-count effect on R1/R2 pooled slates with a proper choice model.
- Different picks per day / time of day if the simulator has context effects (organisers hint time-of-day x weekday).
- Cold users (14.6% of rows, no click history): any signal other than item CTR (none found so far).
- Compare against the organisers' SASRec-lite (0.3955 on this split) with context + tuning (Susan's v1 scored 0.3885).
- Ensemble several half-lives of CTR (expected gain tiny).
