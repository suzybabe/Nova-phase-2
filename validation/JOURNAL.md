Journal full · MD
Team Nova – Validation & Experimentation Journal
Owners: Susan & Lea · Branch: validation

This is our record of every step and decision, with the evidence behind it. It feeds the final presentation (25% of judging): approach, what changed across rounds and why, comparisons, limitations.

Setup log (Sat 14 Nov)
Time	Step	Result
Fri eve	Built validation.py: time-based split (train = older events, test = last days), metrics hit@1 / MRR / known vs new users, experiment log, leaderboard	Tested on synthetic data: random ≈ 0.10 hit@1, popularity ≈ 0.28. The toolkit can tell good models from bad
Fri eve	Built make_practice_data.py (synthetic data in the real format)	Used only to test code before the real data arrived (no training for submission)
10:35	Fetched team repo Nova-phase-2, main tracked from origin	OK
10:36	Created branch validation, committed validation/validation.py	OK
10:37	Push rejected (GH007: private email). Fixed with the GitHub noreply email + git commit --amend --reset-author	Pushed, branch on GitHub
10:47	Briefing rules noted (below)	–
11:04	Downloaded SSH key from team page, chmod 600	OK
11:07	Connected to the VM with SSH (Terminal app, typed manually)	Connected
11:10	Created ~/validation on the VM, copied validation.py with scp	100% copied
11:10	Activated the PyTorch environment on the VM	PyTorch 2.11, GPU available
11:25	Additional rules noted (below)	–
12:06	Lea extended validation.py (rolling folds, diagnostics). Default test window is now 2 days → we always pass --days 1	OK
13:27	SSH connection dropped once → reconnected and now work inside tmux (survives disconnects)	OK
Rules from the briefing
Score per round 0–100; total = sum of 4 rounds (out of 400). Every request counts; missing or out-of-stock = 0.
Score per request = P(our item chosen | a click on the pooled slate of all teams' items). Teams recommending the same item get the same credit.
Final judging: 75% leaderboard + 25% presentation (10 min + 5 min Q&A, max 8 slides, prepare during Round 4).
AI tools allowed on laptops to generate code. AI must not run on the VM, operate SSH, or connect to retrieve VM data. Team members review and run the work.
AI must not run commands on the VM for us; we type every command ourselves.
All training happens on the VM. No training elsewhere and importing models/weights.
Only pre-installed packages on the scoring machine; environment.txt optional (torch by default).
Use verify (sudo check-submission) before every deadline. Only the last upload of a round counts.
How we work
Laptop (VS Code): write/edit code, update this journal, push to GitHub. AI allowed.
VM (Terminal app): run code on the real data and train models. No AI; commands typed by us.
Laptop → VM: scp (typed by us). The VM cannot reach GitHub.
Results → journal: typed in by us.
Git: push to validation any time; one pull request validation → main per round.
How we validate
Time split: train on older feedback, test on the most recent day(s). Only test events with exactly one click are scored.
hit@1 = how often the model ranks the clicked item first among the items actually shown (ties count against the model). MRR gives partial credit.
Rolling check (evaluate_rolling, 2 folds): train day 1 → test day 2, and train days 1–2 → test day 3. A model must win on both.
Limitation: offline hit@1 ranks the 3 items that were shown; the real task picks 1 item from the whole catalogue and is scored against all teams' items.
Round 1 – data released 11:30 · deadline ~15:30
Data facts: 99,998 events (6–8 Jan) · slate size 3 (99.2%) · 25% clicked · 5,000 users (~20 events each) · 75,000 requests (9–13 Jan) · 19% new users in requests · 7,819 catalogue items (92% seen in feedback) Validation split: --days 1 → train 6–7 Jan (66,503 events), test 8 Jan (33,495 events, 4,983 with a click), 0 new users in test Diagnostics:

Data is clean: no duplicates, no missing values, no event with >1 click, clicked item always among shown items.
Strong user fatigue: click rate 50% on a user's first slates → 13% after 15+ slates; daily click rate 39% → 21% → 15%.
Item freshness has no effect on which item is picked (choice rate flat at 0.333 across item ages).
Experiments (single split, --days 1)
Model	Settings	hit@1	MRR	Decision
Random	–	0.3430	0.6168	floor
ClickCount	–	0.4413	0.6781	organisers' example
SmoothedCTR	half_life=1, prior=100	0.5252	0.7362	best simple model → first submission
SmoothedCTR fine-tune	half_life 0.25/0.5/1 × prior 100/300/1000	best 0.5254 (hl=1, p=300)	–	hl=1 optimal, prior insensitive → keep hl=1, p=100
SmoothedCTR no-click weighting (Lea)	no-click weight 1.00 / 0.75 / 0.50 / 0.25	0.5252 / 0.5250 / 0.5230 / 0.5230	0.7362 / 0.7363 / 0.7350 / 0.7350	down-weighting no-clicks doesn't help → keep weight 1
CF item-item + CTR	w=10 / 30 / 100	0.5218 / 0.5248 / 0.5252	–	converges to SmoothedCTR; CF covers ~15% of pairs
CF ALS + CTR	w=1 / 10	0.4096 / 0.5073	–	worse
NovaRanker v1 (Lea)	–	failed	–	-inf scores (catalogue window outside history)
NovaRanker v2 (Lea, LightGBM)	default	0.5071	0.7238	below baseline; fit 86 s
ChoiceLogit (clicked events only)	dim 0/4, reg 1/5, hl=1	0.4889–0.5045	–	worse: ignoring no-click events loses information
COEC (fatigue-adjusted CTR)	half_life 1/3 × prior 2/5/20	best 0.5248 (hl=1, p=5)	0.7359	ties SmoothedCTR → not used
Rolling check (2 windows)
Model	Window 1	Window 2	Average	Decision
SmoothedCTR hl=1, p=100	0.4360	0.5252	0.4734	robust: beats hl=3/p=20 (0.4686) and hl=7/p=20 (0.4664)
MNL blend (Darling), w_mnl=0.7	0.4415	0.5318	0.4794	best: beats SmoothedCTR on both windows
MNL blend, w_mnl=0.9	0.4412	0.5298	0.4784	slightly below 0.7
MNL only, w_mnl=1.0	0.4415	0.5290	0.4782	slightly below 0.7
More training history → higher hit@1 (1 day 0.436 → 2 days 0.525).

**MNL tuning (rolling, w_mnl=0.7):** current settings (LAM=5, HALF_LIFE=1, TIME_BINS=24, LAM_U=0.3) = 0.4794 (best). LAM 1 → 0.4760, LAM 20 → 0.4773, half_life 0.5 → 0.4750, half_life 2 → 0.4729, bins 12 → 0.4786, bins 48 → 0.4785, LAM_U 0.1 → 0.4792, LAM_U 1.0 → 0.4775. Current settings are optimal; no change.

MNL blend (Darling): choice model on the shown slates with a "no click" score per time bin and per user (handles fatigue), blended with log smoothed CTR (0.7 / 0.3). Darling's own holdout: 0.5313 / 0.4412 vs SmoothedCTR 0.5253 / 0.4360; our independent validation matches (0.5318 / 0.4415). On round 1 it picks the same global item as SmoothedCTR for every row. 12h re-click rule (Darling): if the user clicked a live item in the last 12 h, recommend it (re-click rate 21/41 = 51% vs ~28% for the top item). Changes 1,173 of 75,000 rows (1.6%). Can't be measured on the offline split. Lea is testing 3/6/12/24 h windows.

Submission checks
submit branch originally picked the first catalogue item for everyone → replaced with SmoothedCTR; requirements.txt removed; environment.txt → torch.
Submitted ZIP: SmoothedCTR solution.py (hl=1, p=100) with environment.txt = tensorflow (works; torch is the safer default).
A 582-line combined solution (CF + LightGBM, trained inside predict()) is on main: not validated, not used.
Round 1 decision
Currently submitted: SmoothedCTR (half_life=1, prior=100), hit@1 0.5252 / MRR 0.7362, fit < 1 s.
Recommended upload before 15:15: Darling's MNL blend (w_mnl=0.7) + 12h re-click rule, after sudo check-submission says OK. Validated better on both windows (+0.006).
Rejected: CF (converges to SmoothedCTR), ALS (worse), LightGBM ranker (0.5071, 86 s fit), ChoiceLogit (0.5045), COEC (tie), no-click down-weighting (no gain).
Official round 1 score: __ (fill in after 15:30)
What we learned
Recency + strong smoothing of item CTR is a strong signal; a choice model that accounts for "no click" (fatigue) adds a small, consistent gain.
No-click events carry real information: removing or down-weighting them hurts.
Personalisation (CF) and the feature ranker don't help yet with 3 days of data.
Open questions for the mentor
Is offline hit@1 on shown slates a good proxy for the pooled-slate score?
Our model recommends nearly the same item to everyone. Is that sensible, or does the simulator reward personalisation?
Is it OK to paste VM outputs (scores, errors) into an AI on our laptops?
Do the released round results show all teams' items in each slate?
Plan for Rounds 2–4
Re-run validation on each round's data (the released results include the real pooled slates).
Minimal ranker: smoothed CTR + user fatigue first; add features one at a time; keep only those that raise hit@1 on both windows.
Validate any combined solution before it is submitted.
Compare the official round score with the offline hit@1.
Round log template (copy for Rounds 2–4)
Round N – data released at : · deadline :
Data facts: events __ · click rate __ · new-user share __ · catalogue size __

Model	Settings	hit@1	MRR	Decision
Submitted: model __ at : · reason: __ Official round score: __ What we learned / change for next round: __

Evidence for the presentation
Baseline vs best model: ClickCount 0.4413 → SmoothedCTR 0.5252 → MNL blend 0.5318 (window 2)
Biggest single improvement: recency-weighted, smoothed CTR (+8 points over the organisers' example)
What didn't work and why: CF (too little overlap between users), LightGBM ranker (no gain, slow), clicked-only choice model (throws away no-click information)
Limitations: offline hit@1 approximates the real pooled-slate score; new users can't be measured on the time split; re-click rule only estimated

Claude finished the response
