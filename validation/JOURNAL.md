# Team Nova: Validation and Experimentation Journal

**Owners:** Susan and Lea  
**Branch:** `validation`

This journal records our validation work, experiments, results and model decisions. We will also use it when preparing the final presentation.

## Setup log (14 November 2026)

| Time | Step | Result |
|---|---|---|
| Friday evening | Built `validation.py` with chronological splitting, hit@1, MRR, known/new user metrics, experiment logging and leaderboard | Tested on synthetic data. Random scored approximately 0.10 hit@1 and popularity scored approximately 0.28 |
| Friday evening | Built `make_practice_data.py` | Used synthetic data to test the code before receiving the real dataset |
| 10:35 | Fetched the team repository, Nova-phase-2, with main tracked from origin | Complete |
| 10:36 | Created the validation branch and committed `validation/validation.py` | Complete |
| 10:37 | Fixed GitHub push rejection caused by private email settings | Pushed successfully |
| 10:47 | Recorded competition rules | Complete |
| 11:04 | Downloaded SSH key and set permissions using `chmod 600` | Complete |
| 11:07 | Connected to the Huawei VM using SSH | Connected |
| 11:10 | Created `~/validation` on the VM and copied `validation.py` using SCP | Complete |
| 11:10 | Activated the PyTorch environment | PyTorch 2.11, GPU available |
| 11:25 | Recorded additional competition rules | Complete |
| 12:06 | Lea extended `validation.py` with rolling folds and diagnostics | Complete. Use `--days 1` for the single-day validation split |
| 13:27 | SSH connection dropped; reconnected and started using tmux | Complete |

## Competition rules

- Each round is scored from 0 to 100. The total score across four rounds is out of 400.
- Every prediction request counts. Missing predictions or unavailable items receive zero credit.
- The score for a request is based on the probability that our recommended item is chosen, given a click on the pooled slate of all teams' recommendations.
- Teams recommending the same item receive the same credit.
- Final judging consists of 75% leaderboard performance and 25% presentation.
- The presentation is limited to 8 slides, with 10 minutes to present and 5 minutes for questions.
- AI tools may be used on laptops to help generate and review code.
- AI tools must not operate the Huawei VM, execute SSH commands or retrieve VM data.
- Team members must review and run commands themselves.
- All model training must take place on the Huawei VM.
- Models or weights trained elsewhere cannot be imported.
- Only packages available in the scoring environment can be used.
- `environment.txt` is optional, with torch as the default environment.
- Run `sudo check-submission` before each deadline.
- Only the last uploaded submission for a round counts.

## Team workflow

**Laptop**

- Write and review code in VS Code.
- Update documentation.
- Commit and push changes to GitHub.
- Use AI tools for code generation and review where permitted.

**Huawei VM**

- Run experiments on the competition data.
- Train models.
- Execute validation scripts.
- Generate predictions.
- Run submission checks.

Files are transferred from the laptop to the VM using SCP. The VM cannot access GitHub directly.

Experiment results are recorded in the journal and `experiments.csv`.

Changes are pushed to the validation branch and merged into main through pull requests.

## Validation approach

We use chronological validation to avoid training on future interactions.

The model is trained on earlier feedback and evaluated on later events.

Only historical events containing exactly one clicked item are included in the clicked-slate ranking metrics.

**Metrics**

- `hit@1`: The proportion of evaluated slates where the clicked item is ranked first.
- `MRR`: Mean reciprocal rank of the clicked item.
- Known/new user performance: Separate evaluation where the data supports it.

We also use two rolling validation windows:

1. Train on day 1 and test on day 2.
2. Train on days 1 and 2 and test on day 3.

We prefer models that improve performance on both windows.

**Validation limitation**

Historical hit@1 evaluates the model against the three items that were displayed. The actual competition requires selecting one item from the entire catalogue, with scoring based on the pooled slate of team recommendations.

Historical hit@1 is therefore a proxy rather than a direct measurement of the competition score.

---

# Round 1

**Data released:** 11:30  
**Deadline:** Approximately 15:30

## Dataset overview

| Feature | Value |
|---|---|
| Feedback events | 99,998 |
| Feedback period | 6 to 8 January |
| Slate size | 3 items in 99.2% of events |
| Click rate | Approximately 25% |
| Users | Approximately 5,000 |
| Average events per user | Approximately 20 |
| Prediction requests | 75,000 |
| Prediction period | 9 to 13 January |
| New users in prediction requests | Approximately 19% |
| Catalogue items | 7,819 |
| Catalogue items seen in feedback | Approximately 92% |

### Validation split

Using `--days 1`:

- Training period: 6 to 7 January
- Test period: 8 January
- Training events: 66,503
- Test events: 33,495
- Test events with a click: 4,983
- New users in the test set: 0

## Initial diagnostics

The dataset passed the main data quality checks.

- No duplicate feedback events.
- No missing values affecting validation.
- No events containing more than one click.
- Every clicked item was included in the displayed slate.

### User fatigue

Click rates decreased as users received more recommendations.

The click rate was approximately 50% on a user's first slates and decreased to around 13% after more than 15 slates.

Daily click rates also decreased from approximately 39% to 21% and then 15%.

### Item freshness

Item freshness did not show a useful relationship with which item was clicked.

The choice rate remained close to 0.333 across item ages.

---

# Model Experiments

## Single-window validation

All results in this table use `--days 1`.

| Model | Settings | hit@1 | MRR | Decision |
|---|---|---|---|---|
| Random | Default | 0.3430 | 0.6168 | Reference |
| ClickCount | Organisers' example | 0.4413 | 0.6781 | Baseline |
| SmoothedCTR | Half-life 1 day, prior 100 | 0.5252 | 0.7362 | Strongest simple model |
| SmoothedCTR fine-tuning | Half-life 0.25/0.5/1, prior 100/300/1000 | Best 0.5254 | Not recorded | Small improvement, keep prior 100 |
| SmoothedCTR no-click weighting (Lea) | Weights 1.0/0.75/0.5/0.25 | 0.5252/0.5250/0.5230/0.5230 | 0.7362/0.7363/0.7350/0.7350 | No improvement |
| CF item-item + CTR | Weights 10/30/100 | 0.5218/0.5248/0.5252 | Not recorded | Converges to SmoothedCTR |
| CF ALS + CTR | Weights 1/10 | 0.4096/0.5073 | Not recorded | Below baseline |
| NovaRanker v1 (Lea) | Initial version | Failed | Not available | Availability issue |
| NovaRanker v2 (Lea) | LightGBM | 0.5071 | 0.7238 | Below baseline |
| ChoiceLogit | Different dimensions and regularisation | 0.4889 to 0.5045 | Not recorded | Below baseline |
| COEC | Fatigue-adjusted CTR | 0.5248 | 0.7359 | Similar to SmoothedCTR |

## Rolling validation

| Model | Window 1 | Window 2 | Average |
|---|---|---|---|
| SmoothedCTR, half-life 1, prior 100 | 0.4360 | 0.5252 | 0.4734 |
| MNL blend, weight 0.7 | 0.4415 | 0.5318 | 0.4794 |
| MNL blend, weight 0.9 | 0.4412 | 0.5298 | 0.4784 |
| MNL only, weight 1.0 | 0.4415 | 0.5290 | 0.4782 |

The MNL blend with weight 0.7 performed best across both validation windows.

More training history also appeared to improve performance. The single-day training window produced approximately 0.436 hit@1, compared with approximately 0.525 when using two days of training history.

## MNL model and tuning

Darling developed an MNL choice model using the displayed slates and feedback.

The model includes a no-click option with scores that vary by user and time of day.

The final ranking score combines:

- 70% MNL item score
- 30% log smoothed CTR

Darling's own holdout results were:

| Model | Later window | Earlier window |
|---|---|---|
| SmoothedCTR | 0.5253 | 0.4360 |
| MNL blend | 0.5313 | 0.4412 |

Susan's independent rolling validation produced similar results.

### Hyperparameter tuning

Susan tested alternative settings using rolling validation.

| Parameter | Selected value |
|---|---|
| MNL blend weight | 0.7 |
| Item regularisation (LAM) | 5 |
| Half-life | 1 day |
| Time bins | 24 |
| User regularisation (LAM_U) | 0.3 |

The selected settings achieved an average hit@1 of 0.4794.

Other results:

| Setting | Average hit@1 |
|---|---|
| LAM = 1 | 0.4760 |
| LAM = 20 | 0.4773 |
| Half-life = 0.5 | 0.4750 |
| Half-life = 2 | 0.4729 |
| Time bins = 12 | 0.4786 |
| Time bins = 48 | 0.4785 |
| LAM_U = 0.1 | 0.4792 |
| LAM_U = 1.0 | 0.4775 |

The existing settings performed best among those tested.

### Re-click rule

Darling also added a rule that recommends a previously clicked item when:

- The same user clicked it within the previous 12 hours.
- The item is still available at the prediction time.

Darling's analysis found a re-click rate of 21/41, approximately 51%, for the recent-click group examined, compared with approximately 28% for the strongest global item.

The re-click rate declined after 12 hours, which supported using a shorter window.

Lea later ran separate audits to test the re-click window and check how many recommendations actually changed. Those results are documented below.

---

# Lea's Round 1 Experiments

**Date:** 14 November 2026  
**Contributor:** Lea

My work focused on testing alternative ranking methods, fixing validation issues and evaluating the re-click strategy.

The experiments covered LightGBM ranking, no-click impression weighting, re-click window comparison, historical coverage, prediction-time availability and recommendation changes.

## Experiment 1: NovaRanker with LightGBM

### Objective

I wanted to check whether a feature-based ranking model could outperform SmoothedCTR.

I developed NovaRanker using LightGBM and evaluated it using the existing validation setup.

### Initial issue

The first version failed validation because it returned non-finite scores.

The catalogue's `valid_from` and `valid_to` fields covered the future prediction period, while the validation events were from earlier dates.

The model treated every historical candidate as unavailable and returned `-inf` scores.

### Fix

I updated the availability handling so historical validation would not incorrectly exclude every candidate.

The corrected model passed validation and produced finite scores.

### Results

| Model | hit@1 | MRR | Fit time |
|---|---|---|---|
| SmoothedCTR | 0.5252 | 0.7362 | Under 1 second |
| NovaRanker v2 | 0.5071 | 0.7238 | 86 seconds |

NovaRanker performed approximately 1.81 percentage points below SmoothedCTR in hit@1.

It also took considerably longer to fit.

### Conclusion

The additional complexity did not improve performance with the available Round 1 data.

We kept SmoothedCTR as the baseline and did not use NovaRanker for the submission.

The approach may be worth revisiting when more feedback is available.

## Experiment 2: No-click impression weighting

### Objective

I tested whether reducing the influence of impressions without clicks would improve SmoothedCTR.

### Method

I tested four different weights for no-click observations while keeping the remaining SmoothedCTR settings fixed.

### Results

| No-click weight | hit@1 | MRR |
|---|---|---|
| 1.00 | 0.5252 | 0.7362 |
| 0.75 | 0.5250 | 0.7363 |
| 0.50 | 0.5230 | 0.7350 |
| 0.25 | 0.5230 | 0.7350 |

Reducing the no-click weight did not improve hit@1.

### Conclusion

No-click impressions provide useful information.

We kept the original weight of 1.0.

## Experiment 3: Re-click window comparison

### Objective

Darling's model uses a 12-hour re-click window.

I wanted to check whether a shorter or longer window would improve ranking performance.

### Method

I compared the following windows:

- 0 hours
- 3 hours
- 6 hours
- 12 hours
- 24 hours

I used Darling's MNL + SmoothedCTR model and the chronological validation split.

The historical user state was based only on training feedback.

### Validation setup

| Setting | Value |
|---|---|
| Training events | 66,503 |
| Test events | 33,495 |
| Clicked test slates | 4,983 |
| Validation split | `--days 1` |

### Results

| Model | hit@1 | MRR |
|---|---|---|
| SmoothedCTR control | 0.5252 | 0.7362 |
| MNL, 0h | 0.5320 | 0.7396 |
| MNL, 3h | 0.5320 | 0.7396 |
| MNL, 6h | 0.5320 | 0.7396 |
| MNL, 12h | 0.5320 | 0.7396 |
| MNL, 24h | 0.5320 | 0.7396 |

The MNL model improved hit@1 by 0.0068 compared with SmoothedCTR.

However, changing the re-click window did not change the historical validation scores.

This suggested that the historical evaluation was not capturing most of the potential re-click recommendations.

## Experiment 4: Historical re-click coverage audit

### Objective

I investigated why the different re-click windows produced identical validation results.

Historical feedback only contains the items that were displayed.

If a user previously clicked an item that was not included in the next displayed slate, we cannot determine whether they would have clicked it again.

### Method

I used strict training-only click history to avoid using future feedback.

For each window, I counted how often a user had a recent previous click and whether that item appeared in the evaluated slate.

### Results

| Window | Events with recent click | Previous item on slate | Same item clicked | Previous item off-slate |
|---|---|---|---|---|
| 3h | 259 | 0 | 0 | 259 |
| 6h | 1,064 | 1 | 1 | 1,063 |
| 12h | 3,971 | 2 | 2 | 3,969 |
| 24h | 13,828 | 4 | 2 | 13,824 |

For the 12-hour window, 3,971 events had a recent previous click.

Only 2 of those events included the previously clicked item in the displayed slate.

The other 3,969 events did not provide an observable outcome for that item.

### Conclusion

Historical clicked-slate hit@1 cannot reliably measure the additional benefit of the re-click rule.

The identical validation scores do not establish that all re-click windows perform equally well.

We need more outcome information to evaluate the strategy properly.

## Experiment 5: Prediction-window availability audit

### Objective

After checking historical coverage, I wanted to see how many re-click opportunities actually existed during the Round 1 prediction period.

I also checked whether previously clicked items were still available at the requested time.

### Initial issue

The first version of the audit returned:

`Invalid comparison between dtype=str and ndarray`

This happened during the availability comparison.

The audit initially reported zero overrides, but that result was not reliable because the comparison had failed.

### Fix

I corrected the timestamp handling and reran the audit.

I tested two scenarios:

1. Training-only history from the artificial historical split.
2. Full released feedback available at Round 1 prediction time.

### Training-only results

| Window | Eligible live overrides |
|---|---|
| 3h | 0 |
| 6h | 0 |
| 12h | 0 |
| 24h | 0 |

The latest click in the training-only history was approximately 24 hours before the earliest prediction request.

This explained why no eligible overrides were found in that scenario.

### Full released feedback results

| Window | Eligible live overrides | Percentage of requests |
|---|---|---|
| 3h | 71 | 0.09% |
| 6h | 291 | 0.39% |
| 12h | 1,173 | 1.56% |
| 24h | 4,162 | 5.55% |

The latest historical click was approximately 0.01 hours before the earliest prediction request.

With the full released feedback, the current 12-hour window had 1,173 eligible live re-click opportunities.

The 24-hour window had 4,162.

These counts represented eligible candidates, not necessarily changes to the model's original recommendation.

### Conclusion

The corrected audit confirmed that re-click candidates exist during the actual prediction period.

The timestamp issue was in the audit script. It did not establish a problem with Darling's original submission implementation.

I then ran another audit to measure the actual recommendation changes.

## Experiment 6: Re-click recommendation changes audit

### Objective

The previous audit showed that 1,173 requests had eligible re-click candidates within 12 hours.

I wanted to find out how many of these actually changed the item recommended by the MNL model.

I also wanted to check whether Darling's original `user_picks` implementation matched an independent audit.

### Method

I ran `reclick_recommendation_changes_v2.py` on the Huawei VM.

The script used Darling's existing MNL + SmoothedCTR scoring function from `submit-r1/solution.py`.

The experiment used the full released Round 1 feedback and prediction requests.

| Input | Count |
|---|---|
| Prediction requests | 75,000 |
| Catalogue items | 7,819 |
| Historical clicks | 24,892 |

The global MNL + CTR scores were fitted on the Huawei VM.

No holdout labels were used. The experiment measured recommendation changes, not actual click outcomes.

I compared the 3, 6, 12 and 24-hour windows.

### Results

| Window | Eligible overrides | Actual recommendation changes | Percentage changed |
|---|---|---|---|
| 3h | 71 | 70 | 0.09% |
| 6h | 291 | 290 | 0.39% |
| 12h | 1,173 | 1,171 | 1.56% |
| 24h | 4,162 | 4,156 | 5.54% |

Nearly every eligible re-click override changed the original MNL recommendation.

For the 12-hour window, 1,171 of the 1,173 eligible overrides produced a different recommendation.

### Users and items affected

| Window | Distinct users | Distinct items |
|---|---|---|
| 3h | 66 | 66 |
| 6h | 227 | 216 |
| 12h | 758 | 663 |
| 24h | 1,855 | 1,408 |

With the current 12-hour window, the rule changed recommendations for 758 distinct users and selected 663 distinct items.

### Previous click age and MNL score differences

| Window | Median previous click age | Median MNL score gap | 90th percentile score gap |
|---|---|---|---|
| 3h | 2.09h | 0.6443 | 0.8327 |
| 6h | 4.43h | 0.6025 | 0.7989 |
| 12h | 8.38h | 0.6031 | 0.8048 |
| 24h | 16.37h | 0.6158 | 0.8157 |

For the 12-hour window, the median previous click age was 8.38 hours.

The median MNL score gap was 0.6031 in favour of the original recommendation.

This means the re-click rule was usually replacing an item that the global MNL model ranked more highly.

That is expected because the rule uses a separate personalisation signal.

However, the score difference does not prove that the original item would have received more clicks.

### Original implementation check

I compared Darling's original `user_picks` function against the independent corrected audit.

The original implementation used `RECLICK_HOURS=12.0`.

Results:

| Check | Result |
|---|---|
| Overrides returned by original function | 1,173 |
| Missing or incorrect picks | 0 |
| Unexpected picks | 0 |
| Comparison result | PASS |

The original 12-hour re-click choices matched the independent audit exactly.

This confirms that Darling's original re-click implementation was working correctly for the tested prediction inputs.

The earlier timestamp error was specific to the first audit and was not reproduced in this check.

### Findings

The 12-hour rule does change recommendations.

It changes 1,171 of the 75,000 prediction requests, affecting 758 users.

Extending the window to 24 hours would increase the number of changed recommendations to 4,156, affecting 1,855 users.

That is an additional 2,985 changed recommendations compared with the 12-hour window.

However, this audit does not measure whether the additional recommendations are more likely to be clicked.

The historical outcomes of items that were not displayed remain unknown.

### Conclusion

The audit confirmed that:

1. The 12-hour re-click rule activates during the actual prediction period.
2. Almost every eligible override changes the MNL recommendation.
3. Darling's original implementation matches the independent audit.
4. Extending the window to 24 hours would affect considerably more requests.
5. The number of changed recommendations is not enough to determine whether a longer window is better.

## Experiment 7: Final re-click window decision

The 24-hour window would change more recommendations, but Darling's earlier analysis suggested that re-click probability falls to approximately 20% after 12 hours.

This is below the strongest global item's click rate of approximately 28%.

Extending the window could therefore replace stronger global recommendations with less promising personalised ones.

The historical validation data does not provide enough evidence to determine whether those additional overrides would help.

### Decision

Keep Darling's existing 12-hour re-click window for Round 1.

The technical audit passed, and there was no evidence strong enough to justify changing the window before submission.

No submission changes were made as part of these experiments.

## Summary of Lea's work

| Experiment | Result | Decision |
|---|---|---|
| NovaRanker with LightGBM | 0.5071 hit@1, below SmoothedCTR | Not used |
| Historical availability fix | Resolved non-finite validation scores | Fix retained |
| No-click weighting | Lower weights did not improve hit@1 | Keep weight 1.0 |
| Re-click window comparison | Identical historical scores across windows | Investigate coverage |
| Historical coverage audit | Only 2 previous-click items appeared on slate within the 12h group | Historical metric insufficient |
| Prediction-window audit | 1,173 eligible overrides at 12h | Confirmed |
| Recommendation changes audit | 1,171 actual changes at 12h | Confirmed |
| Original implementation comparison | Zero missing, incorrect or unexpected picks | Passed |
| Final window decision | Insufficient evidence for 24h | Keep 12h |

### What I learned

- Historical availability and future prediction availability need to be handled separately.
- More complex models do not necessarily outperform strong baselines.
- No-click events provide useful information.
- Historical slates do not contain enough information to evaluate many alternative recommendations.
- Prediction-time audits can help check whether recommendation rules actually activate.
- Eligible overrides and actual recommendation changes are different measurements.
- A larger number of personalised recommendations does not necessarily mean better performance.
- It is important to check both model behaviour and implementation correctness before changing a submission.

### Next steps

When Round 1 results become available, investigate whether the re-click recommendations improved actual outcomes.

For Round 2, we can also:

- Recheck the MNL model with the additional feedback.
- Compare the submitted model against the official results.
- Evaluate re-click recommendations where outcome information is available.
- Revisit the re-click window if there is evidence that another window performs better.
- Continue testing alternative ranking features individually.

**Status:** All planned Round 1 re-click audits completed. Recommendation-change audit passed. Outcome evaluation pending.

---

# Submission Checks

The `submit` branch originally recommended the first catalogue item for every request.

This was replaced with SmoothedCTR.

Other submission changes included removing `requirements.txt` and updating the environment configuration.

The initial submitted ZIP used:

- Model: SmoothedCTR
- Half-life: 1 day
- Prior strength: 100
- `environment.txt`: tensorflow

The original journal noted that tensorflow worked, although torch was the safer default.

A separate 582-line combined solution using collaborative filtering and LightGBM is present on main. It was not validated or used for the Round 1 submission.

## Round 1 model decision

The initial submitted model was SmoothedCTR.

Following the MNL experiments, the team recommended Darling's MNL blend with the 12-hour re-click rule.

### Selected settings

| Parameter | Value |
|---|---|
| Model | MNL + SmoothedCTR |
| MNL blend weight | 0.7 |
| CTR blend weight | 0.3 |
| Half-life | 1 day |
| Prior strength | 100 |
| Item regularisation | 5 |
| User regularisation | 0.3 |
| Time bins | 24 |
| Re-click window | 12 hours |

The MNL blend performed better than SmoothedCTR on both rolling validation windows.

Lea's additional audits confirmed that the 12-hour re-click implementation worked correctly on the tested prediction inputs.

**Recommended submission:** Darling's MNL blend with the 12-hour re-click rule, subject to `sudo check-submission` passing.

**Final uploaded model:** To be confirmed from the team's submission record.

**Official Round 1 score:** To be added when available.

### Models not selected

- CF item-item: Performance converged to SmoothedCTR.
- CF ALS: Below baseline.
- LightGBM NovaRanker: Below baseline and slower to fit.
- ChoiceLogit: Below baseline.
- COEC: Similar to SmoothedCTR without a clear improvement.
- No-click down-weighting: Did not improve hit@1.

---

# What We Learned From Round 1

SmoothedCTR was a strong baseline and performed substantially better than the organisers' example.

The MNL model improved results further by modelling the no-click option.

Collaborative filtering did not provide enough additional information with the available feedback.

LightGBM did not outperform the simpler models.

No-click weighting experiments showed that retaining the full negative feedback signal was preferable.

The re-click audits confirmed that the 12-hour rule changes recommendations and that Darling's original implementation works correctly.

However, we still do not know whether the re-click overrides improve the actual competition score.

The historical displayed slates are not sufficient to answer that question.

## Open questions for the mentor

1. Is historical hit@1 on displayed slates a useful approximation of the pooled-slate competition score?
2. Is recommending the same strong global item to most users a reasonable strategy for this simulator?
3. Are we allowed to paste VM experiment outputs into AI tools running on our laptops?
4. Will the released results show all items in the pooled slate?
5. Can the released results help evaluate recommendations that were not present in the original historical slates?

---

# Plan for Rounds 2 to 4

1. Run validation again when new feedback is released.
2. Compare the official results against our offline validation scores.
3. Investigate whether the re-click rule improves actual outcomes.
4. Revisit personalisation once more interaction history is available.
5. Test additional ranking features individually.
6. Keep the strongest validated model as the baseline.
7. Record experiment results and reasons for model decisions.
8. Validate any combined solution before submitting it.
9. Run `sudo check-submission` before each deadline.

## Round Log Template

### Round N

**Data released:**  
**Deadline:**  
**Feedback events:**  
**Click rate:**  
**New-user share:**  
**Catalogue size:**

### Experiments

| Model | Settings | hit@1 | MRR | Decision |
|---|---|---|---|---|
| | | | | |

**Submitted model:**

**Submission time:**

**Reason for selection:**

**Official score:**

**What worked:**

**What did not work:**

**Changes for the next round:**

---

# Evidence for the Final Presentation

## Baseline comparison

| Model | hit@1 |
|---|---|
| ClickCount | 0.4413 |
| SmoothedCTR | 0.5252 |
| MNL blend | 0.5318 |

These results are from the same historical validation window.

## Main improvements

- Recency-weighted SmoothedCTR improved hit@1 by approximately 8.4 percentage points over ClickCount.
- The MNL model improved results further by including no-click behaviour.
- The selected MNL settings performed consistently across both rolling validation windows.

## Experiments that did not improve results

- Collaborative filtering
- LightGBM NovaRanker
- Clicked-only choice modelling
- No-click impression down-weighting
- Fatigue-adjusted CTR as a separate model

## Additional validation work

Lea investigated:

- Historical catalogue availability issues.
- LightGBM ranking performance.
- No-click impression weighting.
- Re-click windows from 0 to 24 hours.
- Historical re-click coverage.
- Prediction-time item availability.
- Timestamp comparison errors.
- Actual recommendation changes.
- Original implementation correctness.

## Limitations

- Historical hit@1 does not directly measure the pooled-slate competition score.
- The chronological validation split contains no new users in the test set.
- Most potential re-click recommendations cannot be evaluated using historical displayed slates.
- The number of re-click overrides does not establish their effectiveness.
- The value of personalisation remains uncertain until additional outcome data becomes available.

## Next steps

Use the released Round 1 results to evaluate the submitted model, investigate the performance of re-click recommendations and identify improvements for Round 2.
