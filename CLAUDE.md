# CLAUDE.md

This file provides guidance to Claude Code when working on this repository.

## Project Purpose

Build a production-quality expected goals (xG) model from open football shot data. Two parallel modeling tracks:

- **Track A — "Real xG"**: trained on actual shot outcomes (goal = 1, no goal = 0). This is the primary model.
- **Track B — "Teacher Imitation"**: trained to approximate StatsBomb xG values. This is a distillation / benchmark experiment.

The dataset contains ~131,000 shots from StatsBomb and Wyscout open data, hosted at `https://huggingface.co/luxury-lakehouse/xg-model-statsbomb-wyscout`.

## Relationship to Main Football Analytics Project

This repo is a standalone side project. The main football analytics platform (La Liga / Segunda, Opta data, medallion architecture) lives in a separate repository. Context from that project is in `current_project.md` for reference, but this repo has its own stack, data sources, and conventions.

---

## Stack & Dependencies

| Component | Choice |
|-----------|--------|
| Language | Python 3.11+ |
| Core libs | `pandas`, `numpy`, `scikit-learn`, `xgboost` |
| Plotting | `matplotlib`, `seaborn` (publication-quality defaults) |
| Explainability | `shap` |
| Data format | Parquet for all intermediate/processed datasets |
| Config | YAML files in `configs/` |
| Notebooks | Jupyter, one per phase, for exploration only |
| Scripts | `src/` modules for all reusable pipeline code |

---

## Repository Structure

```
xg-model/
  data/
    raw/                  # Original dataset downloaded from HuggingFace (gitignored)
    interim/              # Intermediate outputs: filtered shots, augmented tables
    processed/            # Final modeling-ready design matrices (train/val/test)
  src/
    data/                 # Data loading, cleaning, sample definition
    features/             # Feature engineering pipeline (baseline + advanced)
    models/               # Model training: logistic regression, XGBoost, teacher
    evaluation/           # Metrics, calibration, subgroup analysis
    inference/            # Inference pipeline and artifact loading
    utils/                # Shared utilities (paths, config loading, plotting helpers)
  notebooks/              # One exploratory notebook per phase (01_audit.ipynb, etc.)
  outputs/
    figures/              # All plots (calibration curves, SHAP, subgroup charts)
    tables/               # CSV metric tables
    models/               # Saved model artifacts (.joblib, .xgb)
    reports/              # Markdown reports per phase
  configs/                # YAML configuration (paths, hyperparameters, feature lists)
  tests/                  # Unit tests for feature engineering and data processing
  current_project.md      # Reference to main football analytics project
  project_plan_v0.md      # Original project brief
```

---

## Phase Workflow

The project follows 12 sequential phases. **Always complete and confirm one phase before starting the next.** At the end of each phase, summarize: what was created, what files were added, what command to run, and what to inspect.

### Phase 1 — Data Audit & Problem Definition

**Goal**: Understand the raw dataset completely before any modeling.

**Theory to explain**:
- Why data auditing is the first step in any ML project (garbage in, garbage out)
- The difference between features and targets, and why defining them explicitly prevents leakage
- What data leakage is, with concrete football examples (e.g., using shot end coordinates, post-shot xG, or goalkeeper position after the shot)
- The concept of information available "at shot release" vs "after shot release" as the leakage boundary

**Tasks**:
- Load dataset, inspect schema, dtypes, nulls, duplicates, row count
- Group columns into: outcome, spatial/location, body-part/technique, contextual, match metadata, freeze-frame/player-location
- Estimate class balance (goals vs non-goals)
- Identify penalty, free kick, header, open-play shot filters
- Flag leakage / post-shot columns and explain why each is forbidden
- Map raw column values (StatsBomb/Wyscout category names, type IDs) to a consistent internal data model — define the mapping dictionary during audit so that all downstream phases reference standardized vocabulary, not provider-specific encodings
- Check whether preferred foot metadata exists per player in the raw data; if not, flag it as a data enrichment need for Phase 2

**Deliverables**: audit script, markdown summary, proposed modeling sample definition, column value mapping dictionary

**Rules**:
- Be conservative about leakage — when in doubt, exclude and explain
- Separate "candidate features" from "forbidden features" with explicit reasoning

---

### Phase 2 — Sample Definition

**Goal**: Create a clean, well-defined modeling dataset.

**Theory to explain**:
- Why sample definition matters (the model can only learn what you show it)
- The case for excluding penalties (near-deterministic, different generative process)
- Whether direct free kicks deserve separate treatment (different shooting geometry)
- What a "shot population" means and why it must be stated explicitly

**Tasks**:
- Apply the column value mapping defined in Phase 1 — transform all raw provider-specific encodings to the internal data model vocabulary
- Filter to open-play shots by default (exclude penalties)
- Decide on direct free kicks (recommend: exclude or flag for separate model)
- Build `shot_id` if none exists
- Enrich with preferred foot per player: if not in raw data, infer statistically (e.g., the foot used on >70% of a player's non-header shots) or join from an external source. Store as a player-level attribute
- Create final one-row-per-shot table
- Save versioned parquet

**Deliverables**: cleaned parquet, data dictionary, inclusion criteria document, preferred foot lookup table

---

### Phase 3 — Feature Engineering

**Goal**: Build a modular, transparent feature pipeline.

**Theory to explain**:
- The bias-variance tradeoff in feature selection (too few = underfitting, too many = overfitting or noise)
- Why spatial features (distance, angle) are the foundation of any xG model — the geometry of shooting
- How to compute shot distance and angle from coordinates (with diagrams/formulas)
- The concept of "visible goal angle" (the angle subtended by the goal from the shooter's position)
- Why freeze-frame features (defender positions, GK position) capture context that coordinates alone cannot
- Feature importance vs feature relevance — a feature can be important in a tree model but irrelevant if it leaks

**Feature groups**:
1. **Baseline spatial**: distance to goal center, distance to near post, distance to far post, visible angle
2. **Baseline categorical**: body part, shot technique, play pattern
3. **Baseline contextual**: under pressure, first-time shot, assist type, home/away, `is_weak_foot` (shot taken with non-preferred foot)
4. **Advanced game-state**: goal difference at time of shot, minute of match
5. **Advanced freeze-frame** (if available): GK distance, GK angle, nearest defender distance, defenders in shooting cone, congestion in box

**Deliverables**: feature builder module, feature list (baseline vs advanced), saved design matrix

**Rules**:
- Start simple, add complexity incrementally
- Make it easy to ablate (remove) features for comparison
- Every feature must pass the "available at shot release" test

---

### Phase 4 — Train / Validation / Test Split

**Goal**: Create a leakage-free, realistic evaluation setup.

**Theory to explain**:
- Why random row-level splitting is wrong for event data (shots from the same match are correlated — the model could memorize match-specific patterns)
- Match-level splitting: all shots from a match go to the same fold
- Time-aware splitting: if multiple seasons exist, train on earlier seasons, validate/test on later ones (simulates real deployment)
- The difference between validation (used for tuning) and test (touched once, for final reporting)
- Why stratification on goal rate matters when the target is imbalanced (~10% goals)
- K-fold cross-validation vs single hold-out: tradeoffs for this dataset size

**Recommended default**: chronological split by season if multiple seasons exist; otherwise match-level grouped split with stratification.

**Deliverables**: split code, split explanation, summary table (shots and goal rate per split)

**Rules**:
- No `match_id` may appear in more than one split — verify this explicitly
- Print split statistics and sanity-check goal rates

---

### Phase 5 — Baseline Model (Logistic Regression)

**Goal**: Establish a principled baseline with an interpretable model.

**Theory to explain**:
- Why logistic regression is the right first model for xG:
  - It directly outputs calibrated probabilities (the sigmoid function maps log-odds to [0,1])
  - Coefficients are interpretable: each coefficient is a log-odds ratio
  - It serves as a "sanity check" — if a tree model can't beat logistic regression, something is wrong
- The logistic function: `p = 1 / (1 + exp(-(b0 + b1*x1 + ... + bn*xn)))`
- Why log loss (cross-entropy) is the natural loss function for probability estimation, not accuracy
- Brier score: mean squared error of probabilities — decomposes into calibration + refinement + uncertainty
- ROC AUC vs PR AUC: discrimination ability, but not sufficient for probability quality
- What a calibration curve / reliability diagram shows: predicted probability bins vs observed frequency
- Why preprocessing matters: scaling numerics for logistic regression, encoding categoricals

**Deliverables**: training script, metrics on val/test, calibration plot, saved predictions

**Rules**:
- Probability quality is the primary objective, not classification accuracy
- Always report Brier score alongside AUC
- Generate coefficient interpretation

---

### Phase 6 — Boosted Tree Model (XGBoost)

**Goal**: Attempt to improve on the baseline with a more flexible model.

**Theory to explain**:
- What gradient boosting is: sequentially fitting trees to the residual errors of the ensemble
- Why XGBoost works well for tabular data: handles non-linearities, interactions, missing values natively
- Key hyperparameters and their intuition:
  - `max_depth`: tree complexity (deeper = more interactions, more overfitting risk)
  - `learning_rate` (eta): step size shrinkage (smaller = more trees needed, but better generalization)
  - `n_estimators` + early stopping: let the data decide when to stop
  - `min_child_weight`: minimum samples in a leaf (regularization)
  - `subsample`, `colsample_bytree`: stochastic regularization (like dropout for trees)
  - `scale_pos_weight`: handling class imbalance
- The `binary:logistic` objective outputs probabilities, but they may not be perfectly calibrated (unlike logistic regression)
- Why early stopping on validation log loss prevents overfitting
- Feature importance: gain vs cover vs frequency; why SHAP values are more reliable

**Deliverables**: training script, tuning summary, metrics comparison vs logistic regression, feature importance, saved model

**Rules**:
- Use early stopping — do not pick `n_estimators` manually
- Keep the pipeline reproducible (set random seeds)
- Do not chase tiny AUC gains if calibration degrades

---

### Phase 7 — Calibration

**Goal**: Ensure predicted probabilities are trustworthy.

**Theory to explain**:
- What calibration means: "when the model says 15% chance of goal, do ~15% of those shots actually go in?"
- Why calibration matters more than discrimination for xG (we sum xG values — miscalibration compounds)
- Reliability diagrams: how to read them, what a perfectly calibrated model looks like (diagonal)
- Brier score decomposition: calibration component + refinement component + uncertainty
- Post-hoc calibration methods:
  - **Platt scaling**: fit a logistic regression on the model's outputs — learns a recalibration sigmoid
  - **Isotonic regression**: non-parametric monotonic recalibration — more flexible but needs more data
  - When each is appropriate (Platt for mild miscalibration, isotonic for severe)
- Why you must calibrate on a held-out calibration set (not training data)
- The "calibration-discrimination tradeoff": aggressive recalibration can sometimes hurt discrimination

**Deliverables**: calibration analysis, reliability plots, recommendation on raw vs calibrated probabilities

**Rules**:
- Calibration is mandatory, not optional
- Always compare before/after calibration with both Brier score and reliability curves
- Make this phase easy to rerun after any model change

---

### Phase 8 — Subgroup Analysis

**Goal**: Understand where the model succeeds and fails.

**Theory to explain**:
- Why global metrics hide subgroup failures (Simpson's paradox in ML)
- The concept of "conditional calibration": is the model calibrated within each subgroup?
- Common xG model failure modes:
  - Headers: harder to predict (body orientation, cross quality not captured)
  - Long shots: low conversion rate makes calibration unstable
  - 1v1s: freeze-frame context is critical
  - Set pieces: different generative process from open play
- How to interpret subgroup metrics: look for systematic over/underestimation patterns

**Subgroups to evaluate**: headers, footed shots, first-time shots, shots from crosses, cutbacks, through balls, counterattacks, crowded-box shots, long-range shots, 1v1 situations

**Deliverables**: subgroup evaluation report, CSV with per-subgroup metrics, analysis of failure modes

---

### Phase 9 — Teacher Imitation Model (Track B)

**Goal**: Build a separate model that approximates StatsBomb xG, not reality.

**Theory to explain**:
- Knowledge distillation: training a student model to mimic a teacher model's outputs
- Why this is regression (predicting a continuous value), not classification
- The philosophical difference: Track A learns from ground truth, Track B learns from a model's opinion
- MSE/RMSE vs MAE: which loss function and why
- What it means when the imitation model fails to match StatsBomb: the gap reveals what features/information StatsBomb uses that we don't have
- Distribution comparison: are our predicted xG values distributed like StatsBomb's?

**Deliverables**: teacher-imitation training pipeline, comparison report (real model vs imitation vs StatsBomb)

**Rules**:
- Keep this completely separate from Track A
- Make it explicit in all outputs that this model approximates a provider estimate

---

### Phase 10 — Comparative Analysis

**Goal**: Answer the key research questions.

**Questions to answer**:
1. How good is my own real xG model compared to a state-of-the-art provider?
2. How close can I get to StatsBomb xG with my features?
3. On which shot types do the two approaches diverge most?
4. Does adding freeze-frame features narrow the gap?
5. Where do missing contextual features likely explain the remaining gap?

**Deliverables**: final comparison report, plots, production recommendation

---

### Phase 11 — Inference Pipeline

**Goal**: Make the model usable on new data.

**Theory to explain**:
- Why an inference pipeline must replicate training preprocessing exactly (feature engineering, scaling, encoding)
- The importance of saving all preprocessing artifacts (scalers, encoders, feature lists)
- Batch inference vs real-time inference tradeoffs

**Deliverables**: inference module, example inference, saved artifacts, usage instructions

---

### Phase 12 — Documentation

**Goal**: Make the project reproducible and explainable.

**Deliverables**: comprehensive README, reproducible run order, caveats, future work

---

## Modeling Conventions

### Leakage Prevention

The fundamental rule: **only use information available at the moment of shot release.**

Forbidden features (post-shot):
- Shot end coordinates (x_end, y_end) — reveal where the ball went after the shot
- GoalMouthY, GoalMouthZ — where the ball crossed the goal line
- Any "blocked" / "saved" / "post" outcome indicators used as features
- xG from another provider used as a feature in Track A (it is the target in Track B)

### Evaluation Hierarchy

1. **Calibration** (Brier score, reliability curve) — most important for xG
2. **Discrimination** (log loss, ROC AUC) — important but secondary
3. **Subgroup behavior** — the model must not systematically fail on common shot types

### Decision Labeling

When there is a modeling choice with tradeoffs, always label it as:
- **Recommended default**: the safest, most principled option
- **Alternative**: a valid option with different tradeoffs
- **Risk**: a choice that could introduce problems

### Reproducibility

- Set random seeds everywhere (`random_state=42` as default)
- Save all artifacts (models, scalers, encoders, feature lists) to `outputs/models/`
- Save all predictions to `outputs/tables/`
- Save all figures to `outputs/figures/`
- Use configs for all tunable parameters

---

## Pedagogical Mode

**This project doubles as a learning exercise.** When working on any phase:

- Explain the theoretical foundation of each technique before implementing it
- Give the mathematical intuition where relevant (formulas, not just names)
- Explain *why* a method works, not just *how* to use it
- When choosing between approaches, explain the tradeoffs in terms of bias/variance, calibration, and interpretability
- Use football-specific examples to ground abstract concepts (e.g., "a shot from 30 meters has low xG because the visible goal angle is small")
- If a concept has a common misconception, flag it (e.g., "high AUC does not mean well-calibrated probabilities")

---

## Working Conventions

- Prefer scripts/modules in `src/` for reusable pipeline code
- Use notebooks in `notebooks/` for exploration and visualization only
- Save all intermediate datasets as parquet in `data/interim/` or `data/processed/`
- Use type hints in all function signatures
- Keep functions short and single-purpose
- Use `configs/` YAML files for paths, hyperparameters, and feature lists — no magic numbers in code
- All plots must be publication-quality: labeled axes, titles, legends, readable font sizes
- File naming: `{phase_number}_{description}.py` or `{phase_number}_{description}.ipynb`

---

## How to Run (will be populated as phases are built)

```bash
# Phase 1: Data audit
python -m src.data.audit

# Phase 2: Sample definition
python -m src.data.sample

# Phase 3: Feature engineering
python -m src.features.build

# Phase 4: Split
python -m src.data.split

# Phase 5: Baseline model
python -m src.models.train_logistic

# Phase 6: XGBoost model
python -m src.models.train_xgboost

# Phase 7: Calibration
python -m src.evaluation.calibration

# Phase 8: Subgroup analysis
python -m src.evaluation.subgroups

# Phase 9: Teacher imitation
python -m src.models.train_teacher

# Phase 10: Comparison
python -m src.evaluation.compare

# Phase 11: Inference
python -m src.inference.predict
```

---

## Current State

> Last updated: 2026-04-26

### Done
- Project plan defined (`project_plan_v0.md`)
- Repository scaffolded
- CLAUDE.md written

### In Progress
- Phase 1: Data Audit (next step)

### Pending
- Phases 2–12
