# Team Nova: Validation and Experimentation Journal

**Owners:** Susan and Lea  
**Working branch:** `validation`  
**Updated:** 14 November 2026  
**Project:** Huawei Tech Arena, NOVA Phase 2

This journal consolidates the team's Round 1 validation history, Lea's independent audits, model decisions, limitations, and the preparation for the mentor's proposed Round 2 Deep & Cross Network (DCN). It is a full journal draft based on the existing GitHub journal and the subsequent experiment outputs. It is not a claim that the final submission or official score has been confirmed.

## 1. Competition rules and working practice

- Four rounds are scored out of 100 each; the final evaluation combines leaderboard performance (75%) and presentation (25%).
- Every prediction request counts. Missing predictions and unavailable recommended items receive zero credit.
- The competition evaluates the probability of choosing the recommended item within the pooled slate of teams' recommendations. Teams recommending the same item receive the same credit.
- Presentation: at most eight slides, ten minutes presenting, five minutes for questions.
- AI assistance may be used on laptops for writing and reviewing code. **AI must not operate the Huawei VM, run SSH or VM commands, or retrieve data from the VM.** Participants execute and review commands themselves.
- Model training must run on the Huawei VM. Do not import weights trained elsewhere. Use packages supported by the scoring environment.
- Run `sudo check-submission` before each deadline. Only the last uploaded submission for a round counts.

**Workflow:** Draft and review scripts on the laptop; keep the GitHub repository and validation journal current; transfer files to Huawei manually; run all competition-data experiments and training on Huawei; copy the resulting terminal outputs into the journal. Changes on the validation branch are merged through pull requests. The Huawei VM is not assumed to have GitHub access.

## 2. Setup and validation framework

### Setup log, 14 November 2026

| Time | Step | Result |
|---|---|---|
| Friday evening | Developed `validation.py` and `make_practice_data.py` | Tested validation on synthetic data |
| 10:35 | Fetched team repository and tracked main | Complete |
| 10:36 | Created validation branch and committed `validation.py` | Complete |
| 10:37 | Resolved GitHub private-email push rejection | Complete |
| 10:47 | Recorded competition rules | Complete |
| 11:04 | Set SSH key permissions | Complete |
| 11:07 | Connected to Huawei VM | Complete |
| 11:10 | Created `~/validation`, transferred scripts and activated torch | Complete |
| 11:25 | Recorded additional competition restrictions | Complete |
| 12:06 | Lea extended validation with rolling folds and diagnostics | Complete |
| 13:27 | Reconnected after SSH interruption and began using tmux | Complete |

### Dataset and evaluation

| Measure | Round 1 value |
|---|---:|
| Feedback events | 99,998 |
| Feedback period | 6–8 January 2025 |
| Displayed items per slate | 3 in approximately 99.2% of events |
| Historical event click rate | Approximately 25% |
| Users in historical feedback | Approximately 5,000 |
| Prediction requests | 75,000 |
| Prediction period | 9–13 January 2025 |
| New users among prediction requests | Approximately 19% |
| Prediction catalogue items | 7,819 |

We use chronological train/test splits and evaluate only historical slates containing exactly one clicked item. **hit@1** is the fraction of clicked slates whose clicked item is ranked first; **MRR** is mean reciprocal rank of the clicked item. The rolling tests are:

- **Fold 1:** 33,233 training events, 6,879 clicked test slates, using the next day as the test window.
- **Fold 2:** 66,504 training events, 4,983 clicked test slates, using the following day as the test window.

Some earlier scripts reported 66,503 training events and 33,495 test events for the single-day split, while later rolling-fold scripts reported 66,504 and 33,494. Preserve each experiment's recorded counts rather than treating these implementations as identical. No new users appeared in the original last-day chronological test population, motivating a separate cold-start diagnostic.

**Important limitation:** These are three-item *shown-slate* ranking tests. The official score involves selecting an item from the available catalogue and evaluating it within a different pooled slate. Historical hit@1 is a useful proxy, not the official competition score; it cannot reveal what would have happened for an item that was not displayed.

### Baseline aggregation clarification

The team reports **0.4794** for the selected MNL blend across two windows. This is a **clicked-slate-weighted pooled hit@1**, not the simple average of 0.4415 and 0.5318. Fold sample sizes differ (6,879 versus 4,983), so the pooled figure is approximately 0.4794. All future DCN comparisons must use the same aggregation and test population.

## 3. Initial diagnostics

The feedback passed key quality checks: no duplicate feedback events, no invalid multiple-click events, and clicked items were included in their displayed slates. Initial diagnostics showed lower overall click rates after repeated user recommendations and across the three feedback days. Item freshness did not show a useful relationship with item choice, with choice rate near one-third across age groups. These descriptive patterns do not by themselves establish a causal fatigue effect or incremental ranking value.

## 4. Team model experiments and selection

### Single-day validation results

| Model | Configuration | hit@1 | MRR | Outcome |
|---|---|---:|---:|---|
| Random | Default | 0.3430 | 0.6168 | Reference |
| ClickCount | Organisers' example | 0.4413 | 0.6781 | Baseline |
| SmoothedCTR | Half-life 1 day, prior 100 | 0.5252 | 0.7362 | Strong simple model |
| SmoothedCTR fine-tuning | Alternative half-lives and priors | Best 0.5254 | Not recorded | Minimal gain |
| No-click weighting (Lea) | Weights 1.0 / 0.75 / 0.5 / 0.25 | 0.5252 / 0.5250 / 0.5230 / 0.5230 | 0.7362 / 0.7363 / 0.7350 / 0.7350 | Keep weight 1.0 |
| CF item-item + CTR | Increasing CTR weight | 0.5218 to 0.5252 | Not recorded | Converged to CTR |
| CF ALS + CTR | Alternative weights | 0.4096 to 0.5073 | Not recorded | Below CTR |
| NovaRanker v1 (Lea) | Initial LightGBM | Failed | N/A | Historical availability bug |
| NovaRanker v2 (Lea) | Corrected LightGBM | 0.5071 | 0.7238 | Below CTR; fit 86 s |
| ChoiceLogit | Alternative regularisation | 0.4889 to 0.5045 | Not recorded | Below CTR |
| COEC | Fatigue-adjusted CTR | 0.5248 | 0.7359 | No clear gain |

### Rolling model comparison

| Model | Fold 1 hit@1 | Fold 2 hit@1 | Pooled weighted hit@1 |
|---|---:|---:|---:|
| SmoothedCTR, half-life 1, prior 100 | 0.4360 | 0.5252 | 0.4734 |
| MNL blend, weight 0.7 | 0.4415 | 0.5318 | **0.4794** |
| MNL blend, weight 0.9 | 0.4412 | 0.5298 | 0.4784 |
| MNL only, weight 1.0 | 0.4415 | 0.5290 | 0.4782 |

Darling developed a multinomial-logit (MNL) choice model over displayed slates with a no-click option and user/time-dependent no-click scores. The item ranking score is **0.7 × MNL item score + 0.3 × log(smoothed CTR)**. The selected settings were half-life 1 day, CTR prior 100, item regularisation 5, user regularisation 0.3, and 24 time bins. Susan's parameter checks found the chosen settings strongest among those tested. The MNL blend outperformed the CTR baseline on both rolling windows.

**Initial submission status:** An earlier submission used SmoothedCTR; the team later recommended Darling's MNL blend with a 12-hour re-click rule. The final uploaded Round 1 model and official score must be confirmed from the team's submission record; neither is asserted here.

## 5. Lea's initial Round 1 investigations

### 5.1 LightGBM NovaRanker and historical availability

The first NovaRanker incorrectly applied the future prediction catalogue's `valid_from`/`valid_to` range to earlier historical feedback, making all historical candidates appear unavailable and causing non-finite scores. Lea corrected the validation availability logic. The revised model reached **0.5071 hit@1 / 0.7238 MRR** with approximately **86 seconds** fitting time, compared with SmoothedCTR's **0.5252 / 0.7362** and under one second fitting. **Decision:** Do not use NovaRanker for Round 1.

### 5.2 No-click impression weighting

Testing no-click weights 1.0, 0.75, 0.5 and 0.25 gave hit@1 values 0.5252, 0.5250, 0.5230 and 0.5230 respectively. **Decision:** Retain full no-click weight 1.0.

### 5.3 Re-click window comparison and historical coverage

The original MNL model uses the user's most recent clicked item from the preceding 12 hours if the item is available. Lea compared windows of 0, 3, 6, 12 and 24 hours. The single-window historical test gave the same MNL hit@1, approximately 0.5320, across all windows. The historical clicked slates rarely contained the earlier clicked item, so the unchanged metric could not establish equivalence of the rules.

| Re-click window | Events with recent click | Earlier clicked item present in shown slate | Earlier clicked item clicked |
|---|---:|---:|---:|
| 3 hours | 259 | 0 | 0 |
| 6 hours | 1,064 | 1 | 1 |
| 12 hours | 3,971 | 2 | 2 |
| 24 hours | 13,828 | 4 | 2 |

### 5.4 Prediction-time eligibility and actual changes

An initial audit failed on a timestamp dtype comparison and produced an unreliable zero count. After fixing the audit, Lea checked full released feedback against the prediction requests and item availability. The original Darling `user_picks` implementation was independently verified and matched the audit exactly, with zero missing, incorrect or unexpected picks.

| Window | Eligible live re-click overrides | Actual recommendation changes | Distinct users affected | Distinct items selected |
|---|---:|---:|---:|---:|
| 3 hours | 71 | 70 | 66 | 66 |
| 6 hours | 291 | 290 | 227 | 216 |
| **12 hours** | **1,173** | **1,171** | **758** | **663** |
| 24 hours | 4,162 | 4,156 | 1,855 | 1,408 |

The median age of the previous click was 8.38 hours for the 12-hour window. The median global MNL score gap favouring the original recommendation was 0.6031. This is a comparison of model scores, not observed click outcomes. Darling's earlier analysis suggested the re-click probability declines after 12 hours, providing a reason not to extend the window solely to generate more changes. **Decision:** Retain 12 hours.

### 5.5 Susan's most-frequent versus most-recent re-click check

Susan found 1,173 eligible re-click requests, including 200 with two or more distinct eligible items, but **zero requests** where most-frequent selection differed from most-recent selection. No user clicked the same item twice within 12 hours, so frequency ties were broken by recency. **Decision:** Keep most recent; reconsider with more history in Round 2.

## 6. Lea's additional cold-start, concentration and personalisation audits

All experiments in this section were executed by the participant on the Huawei VM using the released competition data. They are diagnostic, not submission changes.

### 6.1 Cold-start user holdout

**Script:** `cold_start_test.py`  
**Question:** Does MNL outperform SmoothedCTR when user history is unavailable?

For each chronological fold, randomly withhold 20% of training users with seeds 11, 23 and 37, excluding **all** their training histories before fitting either model. Evaluate cold and warm test-user clicked slates separately. The 12-hour re-click override was not evaluated for cold users. An initial missing-catalogue-item error was fixed using a train-only unseen-item fallback, because some historical item IDs are absent from the future prediction catalogue.

| Group | Mean CTR hit@1 | Mean MNL hit@1 | Mean difference | MNL wins |
|---|---:|---:|---:|---|
| Cold users | 0.4820 | 0.4818 | -0.0002 | 3/6 fold-seed runs |
| Warm users | 0.4694 | 0.4708 | +0.0014 | 5/6 fold-seed runs |

The six runs reuse the same historical folds, so the seed results overlap and are **not independent trials**. Cold/warm slate selection can also differ. **Conclusion:** No consistent MNL advantage for cold users; do not switch models based on this diagnostic. Add cold-start reporting to Round 2 validation.

### 6.2 Top-item dominance

**Script:** `top_item_audit.py`  
**Question:** Do MNL and CTR make meaningfully different global predictions?

The Fold 1 globally top-ranked item was `itm_acfaf7239b0ee9a09046144fe6d73ecf`; Fold 2 and the full-feedback fit favoured `itm_26ec3536f5917e5e62dce7312387e0d5`.

| Measure | Result |
|---|---|
| Fold 1 shown-slate hit@1, CTR / MNL | 0.4351 / 0.4415 |
| Fold 2 shown-slate hit@1, CTR / MNL | 0.5232 / 0.5318 |
| Full-feedback CTR top-1/top-2 score gap | 0.008021 |
| Full-feedback MNL top-1/top-2 score gap | 0.040247 |
| Global-only CTR predictions | Same top item in 75,000/75,000 requests |
| Global-only MNL predictions | Same top item in 75,000/75,000 requests |
| CTR/MNL global-only disagreements | **0/75,000** |
| Requests with no available item | 0 |
| Final original recommendations retaining the global top item after 12h overrides | **73,829/75,000 (98.44%)** |

In Fold 2, the future globally leading item appeared in only 25 historical test slates and received 7 clicks (28% observed rate). This small exposure count cannot prove it is the optimal recommendation for every future request.

**Conclusion:** Although MNL improves historical within-slate ranking, MNL and CTR produce identical **global-only** Round 1 predictions. The existing 12-hour override accounts for the 1,171 observed changes. Future candidate models must be audited for actual prediction differences, not just offline ranking metrics.

### 6.3 Older-click personalisation

**Script:** `personalized_click_audit.py`  
**Question:** Do user clicks from 12–72 hours earlier improve ranking beyond the existing 12-hour override?

Tested recency-decayed user-item boosts with a 48-hour half-life and strengths 0.25, 0.5, 1.0 and 2.0. Historical comparisons used training-only click histories; prediction comparisons preserved the original 12-hour override.

| Boost alpha | Fold 1 hit@1 change (no training-only 12h candidate) | Fold 2 change | Changed prediction requests | Share changed |
|---|---:|---:|---:|---:|
| 0.25 | +0.00037 | +0.00071 | 441 | 0.59% |
| 0.50 | +0.00019 | +0.00071 | 2,789 | 3.72% |
| 1.00 | +0.00000 | +0.00047 | 13,480 | 17.97% |
| 2.00 | +0.00000 | +0.00024 | 23,006 | 30.67% |

Only **16 of 6,879** clicked historical slates in Fold 1 and **23 of 4,983** in Fold 2 contained an item eligible for the older-click boost. Yet **23,419 of 75,000** prediction requests had older-click candidates. The small historical coverage cannot support confident claims about thousands of future changed recommendations. **Decision:** No Round 1 change; revisit with richer Round 2 feedback.

## 7. Mentor's DCN proposal and feature diagnostics

Susan relayed the mentor's suggestion to test a **Deep & Cross Network (DCN)** in Round 2, starting with **2–3 cross layers** and candidate features including smoothed CTR, MNL score, user fatigue, time of day and CF score. The ML team is responsible for implementation and training on Huawei. Lea's independent contribution is to test feature quality and evaluate model ablations and prediction behaviour.

### 7.1 User fatigue and time-of-day descriptive diagnostic

**Script:** `dcn_feature_diagnostic.py`  
**Design:** Analyse held-out impression CTR, using training-only prior same-user-item exposures and six-hour UTC time buckets. The script does **not** train a DCN or estimate incremental DCN uplift.

| Fatigue coverage | Fold 1 | Fold 2 |
|---|---:|---:|
| Held-out impressions | 99,531 | 100,234 |
| Prior same-user-item exposure in previous 72h, training history only | 359 | 731 |
| Coverage | **0.36%** | **0.73%** |

There was no stable directional fatigue effect across folds. This metric excludes exposures arising during the held-out window itself, so it does not fully measure real-time fatigue.

| UTC six-hour bucket | Fold 1 impression CTR | Fold 2 impression CTR |
|---|---:|---:|
| 00:00–05:59 | 0.0842 | 0.0556 |
| 06:00–11:59 | 0.0725 | 0.0483 |
| 12:00–17:59 | 0.0618 | 0.0498 |
| 18:00–23:59 | 0.0582 | 0.0451 |

The highest observed CTR occurred in the first UTC bucket and the lowest in the last in both folds. **Caution:** Predicting whether a user clicks *anything* is different from ranking *which item* the user chooses. The descriptive pattern does not establish incremental ranking value or causality.

### 7.2 Incremental item-by-time ranking test

**Script:** `time_of_day_incremental_test.py`  
**Question:** Do train-only item-specific time-of-day CTR adjustments improve Darling's MNL ranking?

The method adds smoothed item-by-six-hour-bucket CTR residuals to MNL item scores. Four configurations varied adjustment strength and prior. The initial script produced a duplicate-index mapping error; the lookup was fixed by deduplicating item IDs, and the corrected script ran successfully on Huawei.

**Baseline:** Fold 1 MNL hit@1 0.4415, MRR 0.6805; Fold 2 hit@1 0.5318, MRR 0.7396.

| Alpha | Prior | Fold 1 hit@1 change | Fold 2 hit@1 change | Round 1 prediction changes versus original final recommendations |
|---|---:|---:|---:|---:|
| 0.5 | 25 | -0.01032 | -0.01264 | 36,940 (49.25%) |
| 1.0 | 25 | -0.01962 | -0.02328 | 36,940 (49.25%) |
| 0.5 | 100 | -0.00625 | -0.00682 | 18,097 (24.13%) |
| 1.0 | 100 | -0.00611 | -0.00743 | 36,940 (49.25%) |

**MRR also decreased in every configuration in both folds.** The weakest degradation was alpha 0.5 with prior 100, but it still reduced hit@1 by approximately 0.63 and 0.68 percentage points. Some configurations changed almost half the predictions while performing worse on historical shown slates.

**Conclusion:** Do not add the simple item-time residual boost. This is a negative result for **this feature construction**, not proof that a DCN cannot use time-of-day interactions. Avoid repeatedly tuning the same folds; require a DCN with/without-time ablation.

## 8. Consolidated experiment decisions

| Investigation | Main finding | Decision |
|---|---|---|
| NovaRanker LightGBM | 0.5071 hit@1 versus CTR 0.5252 | Do not use |
| No-click down-weighting | No improvement | Keep weight 1.0 |
| Re-click implementation | 1,173 eligible, 1,171 actual changes, independent check passed | Keep original 12h rule |
| Most frequent vs most recent | Zero changed choices | Keep most recent |
| Cold-start user holdout | MNL minus CTR mean -0.0002 for cold users | No model switch |
| Global item dominance | MNL and CTR global-only recommendations identical for all 75,000 requests | Audit actual prediction changes |
| Older-click affinity | Tiny historical gains on only 39 affected shown slates | No submission change |
| Fatigue diagnostic | 0.36%/0.73% training-history coverage | Experimental DCN feature |
| Time-of-day descriptive analysis | Hourly CTR pattern across both folds | Test incremental ranking value |
| Item-by-time residual boost | Every tested configuration reduced hit@1 and MRR on both folds | Reject this simple boost |

**Round 1 model recommendation remains:** Darling's MNL + smoothed CTR blend (0.7 / 0.3), half-life 1 day, prior 100, 12-hour most-recent re-click rule. This journal does **not** confirm the final uploaded file or official Round 1 score.

## 9. Round 2 DCN validation plan and responsibilities

**ML team:** Implement and train DCN on Huawei with a modest 2–3 cross-layer architecture and agreed candidate features.  
**Susan:** Coordinate shared validation and baseline comparisons.  
**Lea:** Lead feature diagnostics, leakage checks, ablation comparisons, cold-start reporting and prediction-change audits, coordinating to avoid duplicate work.

### DCN evaluation checklist

- [ ] Confirm the **0.4794 pooled hit@1** calculation, test populations and exact chronological splits.
- [ ] Use the same fold boundaries and scoring definitions for MNL and DCN; report each fold's hit@1 and MRR as well as pooled weighted hit@1.
- [ ] Verify that features, encoders, user histories, item statistics and preprocessing use only information available at the prediction timestamp. No test labels or future impressions in training.
- [ ] Run DCN with and without **time-of-day** features.
- [ ] Run DCN with and without **user fatigue** features, checking exposure coverage.
- [ ] If included, test the incremental value of **CF score** rather than assuming it helps.
- [ ] Report **cold-start versus warm-user** metrics and subgroup sample sizes.
- [ ] Compare actual available-item predictions against the original MNL + 12-hour override, including how many requests and users change.
- [ ] Check runtime, model complexity and consistency across both validation windows.
- [ ] Run all fitting on Huawei; validate any candidate submission using the required competition checks.

### Pending items

- Susan to confirm the agreed baseline protocol and any differences from the existing journal's 0.4794 weighted score.
- ML team to provide the DCN implementation, feature definitions and reproducible evaluation entry point.
- Official Round 1 leaderboard score and final uploaded submission status to be added when confirmed.
- Re-evaluate personalised ranking and re-click choices when Round 2 feedback provides more coverage.

## 10. Evidence for the final presentation

**Core narrative:** A strong recency-smoothed CTR baseline was improved by an MNL choice model on both chronological historical folds. Several more complex or personalised alternatives did not demonstrate robust improvements. Validation went beyond hit@1 to audit catalogue availability, historical coverage, actual prediction changes, cold-start behaviour and implementation correctness.

**Key numbers:** ClickCount 0.4413; SmoothedCTR 0.5252; MNL 0.5318 on the later historical window. Pooled rolling MNL hit@1 0.4794. Original re-click rule changes 1,171 of 75,000 requests. MNL and CTR make identical global-only Round 1 recommendations. Simple item-time boosting reduces ranking performance despite changing many predictions.

**Limitations to state explicitly:** Historical shown-slate ranking differs from the official pooled-slate metric; off-slate counterfactual clicks are unavailable; user-fatigue history coverage is low; repeated random holdout seeds are correlated; descriptive time-of-day CTR is not proof of ranking uplift; the official submission outcome is not yet recorded.

**Next milestone:** Validate the ML team's DCN candidate using matched chronological folds and controlled feature ablations, then update the journal with the actual results and final decision.

---

**Journal status, 14 November 2026:** Round 1 diagnostic experiments and preliminary DCN feature tests documented. No submission files changed by Lea's additional audits. DCN training and official Round 1 score remain pending confirmation.
