# Team Nova — Huawei Tech Arena Ireland 2026 (Phase 2 Final)

**Author:** Susan Ogozi — Year 3 BSc Computing Science, Griffith College Dublin
**Team:** Team Nova (6 members) | **Role (Round 3):** Model Improvement | **Branch:** `phase-3`
**Competition:** Intelligent Marketplace: Generative Sequential Recommendation Under Constraints
**Target:** Top 6 — we need ~2.4 pts/round more than 6th place in Rounds 3 & 4 (assuming teams repeat their R2 scores; R3 & R4 count double).

---

## 1. Competition Overview & Scoring
- **Objective:** One valid catalogue item per request (`400,000` requests, `20,000` users, 9–13 Jan).
- **Score:** All teams' picks are pooled into one slate per request; score = `100 × average probability` the user picks our item.
- **Final ranking:** Weighted round average (R1 1x, R2 1x, R3 2x, R4 2x). Top 6 present at 17:00 (75% leaderboard + 25% presentation; ≤8 slides, 10 min + 5 min Q&A).
- **Standing after R2:** 13th (R1 17.49, R2 48.89).

---

## 2. Round 3 Data Shift
| Property | Rounds 1 & 2 | Round 3 |
| :--- | :--- | :--- |
| **Feedback** | ~175k events (R2) | 2M events (6–8 Jan), 20,000 users, 3-item random slates, 5.5% click rate |
| **Requests** | 75,000 | 400,000 (20 per user, 9–13 Jan) |
| **Repeat behaviour** | Previously clicked items chosen ~30% vs ~13% for others | Re-clicks only **0.3%** (idea to test: mask past clicks) |
| **Popularity** | Strong head items | Flat (**top-100 items = 5%** of clicks) |
| **Main signal** | User habit (re-clicks) | Sequence: most common next item = **21%** of next clicks; organisers hint **time-of-day × weekday** context |

---

## 3. Repository & VM Structure
```text
susan/r3/                       # GitHub (branch: phase-3)
├── solution.py                 # Submission: predict(data_dir, batch_size) — must keep this exact name
├── r3_eval.py                  # Offline validation: train 6–7 Jan, test on 8 Jan slates (hit@1, MRR, catalog MRR)
└── README.md                   # This file

VM only (never commit to GitHub):
├── ~/data/round-3/             # Competition dataset
├── ~/susan_r3/starter          # Organisers' SASRec-lite baseline
└── ~/validation/check_solution.py  # Rules, schema & runtime checker
```

---

## 4. Results (validation: train 6–7 Jan, test 8 Jan)
| Version | Slate hit@1 | MRR | Catalog MRR | Runtime (400k rows) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Starter (organisers) | – | – | – | – | pending |
| v1 (context + bigger + cold start) | – | – | – | – | pending |