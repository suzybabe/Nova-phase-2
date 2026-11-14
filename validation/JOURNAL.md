# Team Nova – Validation & Experimentation Journal
Owners: Susan & Lea · Branch: `validation`

This is our record of every step and decision, with the evidence behind it.
It feeds the final presentation (25% of judging): approach, what changed across rounds and why, comparisons, limitations.

---

## Setup log (Sat 14 Nov)

| Time | Step | Result |
|---|---|---|
| Fri eve | Built `validation.py`: time-based split (train = older events, test = last 5 days), metrics hit@1 / MRR / known vs new users, experiment log, leaderboard | Tested on synthetic data: random ≈ 0.10 hit@1, popularity ≈ 0.28. The toolkit can tell good models from bad |
| Fri eve | Built `make_practice_data.py` (synthetic data in the real format) | Used only to test code before the real data arrived (no training for submission) |
| 10:35 | Fetched team repo `Nova-phase-2`, `main` tracked from origin | OK |
| 10:36 | Created branch `validation`, committed `validation/validation.py` (248 lines) | OK |
| 10:37 | Push rejected (GH007: private email). Fixed with the GitHub noreply email + `git commit --amend --reset-author` | Pushed, branch on GitHub |
| 10:47 | Briefing rules noted (below) | – |
| 11:04 | Downloaded SSH key from team page, `chmod 600` | OK |
| 11:07 | Connected to the VM with SSH (Terminal app, typed manually) | Connected |
| 11:10 | Created `~/validation` on the VM, copied `validation.py` with `scp` | 100% copied |
| 11:10 | Activated the PyTorch environment on the VM | PyTorch 2.11, GPU available |
| 11:25 | Additional rules noted (below) | – |

### Rules from the briefing
- Score per round 0–100; total = sum of 4 rounds (out of 400). Every request counts; missing or out-of-stock = 0.
- Teams recommending the same item get the same credit.
- Final judging: **75% leaderboard + 25% presentation** (10 min + 5 min Q&A, max 8 slides, prepare during Round 4).
- AI tools allowed **on laptops** to generate code. AI must **not** run on the VM, operate SSH, or connect to retrieve VM data. Team members review and run the work.
- AI must not run commands on the VM for us; we type every command ourselves.
- All training happens on the VM. No training elsewhere and importing models/weights.
- Use `verify` (check-submission) before every deadline.

### How we work
- **Laptop (VS Code):** write/edit code, update this journal, push to GitHub. AI allowed.
- **VM (Terminal app):** run code on the real data and train models. No AI; commands typed by us.
- **Laptop → VM:** `scp` (typed by us). The VM cannot reach GitHub.
- **Results → journal:** typed in by us.

---

## Round log (copy this block for each round)

### Round N – data released at __:__ · deadline __:__
**Data facts** (from `explore.py`): events __ · click rate __ · new-user share __ · catalogue size __

**Experiments** (from `experiments.csv`, best first)
| Model | Settings | hit@1 | MRR | hit@1 new users | Decision |
|---|---|---|---|---|---|
| | | | | | |

**Submitted:** model __ at __:__ · reason: __
**Official round score:** __
**What we learned / change for next round:** __

---

## Evidence for the presentation
- Baseline vs final model: hit@1 __ → __
- Biggest single improvement: __ (round __)
- What didn't work and why: __
- Limitations: offline hit@1 is an approximation of the real pooled-slate score; __
## Round 1 – data released 11:30 · deadline ~15:30
**Data facts:** 99,998 events (6–8 Jan) · slate size 3 · 25% clicked · 5,000 users (~20 events each) · 19% new users in x_pred · 7,819 items
**Validation split:** --days 1 → train 6–7 Jan (66,503 events), test 8 Jan (33,495 events, 4,983 with a click), 0 new users in test
**Finding:** strong user fatigue: click rate 50% (first slates) → 13% (15th–20th slate); daily click rate 39% → 21% → 15%

| Model | Settings | hit@1 | MRR | Decision |
|---|---|---|---|---|
| Random | – | 0.3430 | 0.6168 | floor |
| ClickCount | – | 0.4413 | 0.6781 | organisers' example |
| **SmoothedCTR** | half_life=1, prior=100 | **0.5252** | **0.7362** | **best → submitted** |
| CF item-item + CTR | w=10 / 30 / 100 | 0.5218 / 0.5248 / 0.5252 | – | converges to SmoothedCTR; CF covers ~15% of pairs |
| CF ALS + CTR | w=1 / 10 | 0.4096 / 0.5073 | – | worse |
| NovaRanker v1 (Lea) | – | failed | – | -inf scores (catalogue window outside history) |
| NovaRanker v2 (Lea, LightGBM) | default | 0.5071 | 0.7238 | below baseline; fit 86 s |

**Submission fix:** submit branch was picking the first catalogue item → replaced with SmoothedCTR; removed requirements.txt; environment.txt = torch.
**What we learned:** recency + strong smoothing of item CTR is the best signal so far. Personalisation (CF) and the feature ranker don't help yet with 3 days of data.
