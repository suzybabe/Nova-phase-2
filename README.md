# NOVA | Huawei Tech Arena 2026

**Phase 2 — Adaptive Recommendation Challenge**

Team NOVA is participating in the Huawei Tech Arena 2026 Ireland Finals, developing a data-driven recommendation system that selects the most relevant available item for each user request.

Our approach combines recommendation modelling, candidate selection, validation, experimentation, and iterative optimisation across four competition rounds.

## Team & Responsibilities

| Team Member | Technical Area | Responsibility |
|---|---|---|
| Darling | Area 1 | Data Exploration & Feature Engineering |
| Ahmed | Area 2 | Recommendation Modelling |
| Addy | Area 3 | Candidate Selection & Ranking |
| Lea Stanisavljevic | Area 4 | Validation & Experimentation |
| Susan Ogozi | Area 4 | Validation & Experimentation |
| Chioma | Area 5 | Integration & Submission |

All team members contribute collaboratively to model development, testing, optimisation, and final decision-making.

## Competition Objective

Build a recommendation agent that:

- Receives a user ID and timestamp.
- Selects one valid, available item per request.
- Competes against recommendations from other teams in a shared simulator.
- Learns from interaction history and feedback between rounds.
- Maximises the probability that its recommended item is selected.

The competition consists of four scored rounds, followed by a technical presentation for qualifying finalists.

## Technical Workflow

**1. Data & Features — Darling**

Explore user interactions, item availability, temporal patterns, and features that may improve recommendation quality.

**2. Recommendation Models — Ahmed**

Develop and optimise models that estimate user-item relevance using historical interaction data.

**3. Candidate Selection — Addy**

Develop efficient candidate-generation and ranking strategies while respecting item availability constraints.

**4. Validation & Experimentation — Lea & Susan**

Build offline evaluation pipelines, establish baselines, compare experiments, identify weaknesses, and monitor performance across user cohorts.

**5. Integration & Submission — Chioma**

Integrate the selected approach into the required prediction interface, verify submission compatibility, and coordinate packaging and testing.

## Repository Workflow

Each technical area works on its assigned branch. Changes should be committed to the relevant branch and reviewed before integration.

- Keep `main` stable.
- Coordinate before merging branches.
- Do not create pull requests until the team agrees.
- Do not commit SSH keys, team login links, credentials, or private competition data.
- Use separate working directories on the shared H100 VM.
- Coordinate all official submissions to avoid accidental replacement.

## Evaluation

The official score is based on the probability that the team's recommendation is selected from the pooled recommendation slate.

Offline validation is used to compare approaches and diagnose model behaviour, but offline metrics are not equivalent to the official simulator score.

The final competition result combines:

- **75%** — Competition leaderboard performance
- **25%** — Technical presentation

## Technology

Python · PyTorch · Pandas · NumPy · Git · GitHub · NVIDIA H100

## Competition Environment

Model training and testing take place on the designated Huawei H100 virtual machine.

The competition requires a submission archive containing `solution.py`, necessary model artifacts, and optionally `requirements.txt`.

AI-assisted development is permitted locally, but AI agents must not autonomously operate the competition VM or SSH connection.

---

**Team NOVA | Huawei Tech Arena Ireland 2026**

*Collaborative development. Evidence-driven experimentation. Adaptive recommendation.*
