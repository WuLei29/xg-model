# CLAUDE.md

This file provides guidance to Claude Code when working on this repository.

## Project Purpose

Build a production-quality expected goals (xG) model from open football shot data.

- **Primary model**: trained on actual shot outcomes (goal = 1, no goal = 0).
- **Benchmark**: StatsBomb xG values are used as a reference to evaluate and understand our model — not as a training target.

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
    models/               # Model training: logistic regression, XGBoost
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

### Phase 9 — Comparative Analysis & Gap Investigation

**Goal**: Benchmark our model against StatsBomb xG, understand where they agree and diverge, and identify what contextual information we are missing.

**Theory to explain**:
- Why StatsBomb xG is not ground truth — it is another model's opinion, trained on richer features (freeze frames, GK position, etc.). Treating it as a benchmark is valid; treating it as a calibration target would inherit their biases.
- What "gap analysis" means: systematic over/underestimation by shot type reveals which situations our feature set cannot explain.
- Simpson's paradox in model comparison: a model can have better global metrics but worse behavior on important subgroups.
- Why comparing calibration curves across models is more informative than comparing AUC alone.

**Tasks**:
1. **Three-way benchmark on test set**: evaluate LR, XGBoost, and StatsBomb xG with the same metrics (Brier score, log loss, ROC AUC, calibration curve). StatsBomb xG is treated as a "model" — its predictions are already in the dataset as `statsbomb_xg`.
2. **Prediction distribution comparison**: plot histogram of predicted xG values for each model. Are they similarly distributed? Do they agree on shot difficulty?
3. **Divergence analysis by subgroup**: compute mean(our_xg) − mean(statsbomb_xg) per subgroup (distance bins, headers, body part, play pattern, assist type). Identify the largest gaps and explain them in terms of missing features.
4. **Scatter plot**: our XGBoost xG vs StatsBomb xG per shot — color by shot type. Identify systematic clusters of disagreement.
5. **Stacking experiment** (research extension): train a second-stage logistic regression that takes our XGBoost output + `statsbomb_xg` as inputs, trained on actual outcomes. This answers: "what do we gain if StatsBomb is available at inference time?" Frame explicitly as a research exercise — this model is not deployable without StatsBomb.
6. **Gap interpretation report**: for each major divergence, explain the most likely cause (missing freeze-frame, GK position, cross quality, body orientation not captured).

**Theory to explain — stacking**:
- What stacking (model blending) is: a meta-learner trained on the outputs of base models
- Why our model and StatsBomb can have orthogonal errors (they were trained on different features / data)
- The deployment constraint: the stacked model requires StatsBomb xG at inference time — it cannot be used standalone

**Questions to answer**:
1. How good is our XGBoost model vs StatsBomb on the same test set?
2. On which shot types do the two approaches diverge most?
3. Where do missing contextual features most likely explain the remaining gap?
4. What do we gain from stacking, and is it worth the dependency on StatsBomb?

**Deliverables**: `src/evaluation/compare.py`, three-way metrics table, prediction distribution plots, divergence heatmap by subgroup, scatter plot, stacking experiment results, `09_compare_report.md`

**Rules**:
- StatsBomb xG is a benchmark, never a calibration target for our primary model
- All three "models" must be evaluated on the same test split
- The stacking experiment must be clearly labelled as requiring StatsBomb at inference time

---

### Phase 10 — Production Recommendation

**Goal**: Synthesise all findings into a clear recommendation about which model to use and under what conditions.

**Tasks**:
- Summarise the full model comparison: LR vs XGBoost vs StatsBomb vs stacked
- State the recommended production model with justification (expected: calibrated XGBoost full-feature model)
- List the known failure modes and limitations
- Describe what additional data or features would most improve the model
- Document the deployment conditions (standalone vs with StatsBomb available)

**Deliverables**: `10_production_recommendation.md` (part of the Phase 11 inference pipeline documentation)

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
- `statsbomb_xg` as a feature in the primary model — it is a benchmark reference, not an input. It may only be used in the explicitly-labelled stacking experiment in Phase 9.

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

# Phase 9: Comparative analysis & gap investigation
python -m src.evaluation.compare

# Phase 11: Inference
python -m src.inference.predict
```

---

## Current State

> Last updated: 2026-04-30

### Done
- Project plan defined (`project_plan_v0.md`)
- Repository scaffolded
- CLAUDE.md written
- **Phase 1: Data Audit** — `src/data/audit.py` · outputs: `01_audit_report.md`, `01_value_mapping.json`, `01_null_summary.csv`
- **Phase 2: Sample Definition** — `src/data/sample.py` · outputs: `02_open_play_shots.parquet` (82,380 shots, 10.3% goal rate), `02_free_kick_shots.parquet` (4,229 shots, 6.5% goal rate), `02_preferred_foot.parquet`, `02_sample_report.md`, `02_sample_stats.csv`
- **Phase 3: Feature Engineering** — `src/features/build.py` · `configs/features.yaml` · outputs: `03_design_matrix.parquet` (82,380 rows, 22 model features), `03_feature_list.json`, `03_feature_report.md`
- **Phase 4: Train/Val/Test Split** — `src/data/split.py` · `configs/split.yaml` · outputs: `04_train.parquet`, `04_val.parquet`, `04_test.parquet`, `04_split_index.parquet`
- **Phase 5: Baseline Model (Logistic Regression)** — `src/models/train_logistic.py` · outputs: `05_logistic_baseline.joblib`, `05_logistic_full.joblib`, `05_val_predictions.parquet`, `05_test_predictions.parquet`, `05_metrics.csv`, `05_coefficients.csv`, `05_calibration_curve.png`, `05_roc_curve.png`, `05_logistic_report.md`
- **Phase 6: XGBoost Model** — `src/models/train_xgboost.py` · `configs/xgboost.yaml` · outputs: `06_xgboost_baseline.json`, `06_xgboost_full.json`, `06_val_predictions.parquet`, `06_test_predictions.parquet`, `06_metrics.csv`, `06_shap_summary.png`, `06_shap_bar.png`, `06_calibration_curve.png`, `06_roc_curve.png`, `06_xgboost_report.md`

### Key decisions made
- StatsBomb data only (drops Wyscout — richer features, consistent schema, statsbomb_xg available as benchmark)
- Open-play model: excludes penalties, direct free kicks, corners, kick-offs
- Free-kick model: separate dataset, logistic regression only (sample too thin for XGBoost)
- No freeze-frame features — production event data does not have them
- Play pattern included: grouped into 3 binary flags (regular, counter, set_piece_restart); other restarts are implicit baseline
- Preferred foot inferred statistically (≥70% threshold); `is_weak_foot` added as feature
- Feature sets: "baseline" (20 features: spatial+categorical+contextual), "full" (22: adds score_diff_at_shot, is_late_game)
- Coordinate system: SPADL standard (105x68 metres). Raw StatsBomb coordinates (120x80) are converted in Phase 2 (sample.py). All spatial features are in metres.
- All features recomputed from raw coordinates (distance_to_goal, visible_angle, etc.) — not relying on StatsBomb precomputed values
- XGBoost config: `max_depth=4`, `lr=0.05`, `min_child_weight=15`, `subsample/colsample_bytree=0.8`, early stopping on val log loss (50 rounds patience)
- XGBoost models saved as native JSON (portable, human-readable); LR models saved as joblib pipelines
- SHAP via `TreeExplainer` on val sample (5000 rows); beeswarm + mean|SHAP| bar chart

- **Phase 7: Calibration** — `src/evaluation/calibration.py` · outputs: `07_metrics.csv`, `07_test_predictions.parquet`, `07_val_predictions.parquet`, calibrated model artifacts (Platt + isotonic variants for LR and XGBoost), `07_calibration_report.md`
- **Phase 8: Subgroup Analysis** — `src/evaluation/subgroups.py` · outputs: `08_subgroup_metrics_lr.csv`, `08_subgroup_metrics_xgb.csv`, `08_subgroup_report.md`
- **Phase 9: Comparative Analysis & Gap Investigation** — `src/evaluation/compare.py` · outputs: `09_benchmark_metrics.csv`, `09_divergence_by_subgroup.csv`, `09_stacking_metrics.csv`, `09_calibration_comparison.png`, `09_prediction_distributions.png`, `09_scatter_xgb_vs_statsbomb.png`, `09_divergence_by_subgroup.png`, `09_stacking_calibration.png`, `09_compare_report.md`
- **Phase 10: Production Recommendation** — `outputs/reports/10_production_recommendation.md`

### Key findings (Phase 9)
- StatsBomb xG: Brier=0.0754, AUC=0.8143 vs our XGBoost: Brier=0.0801, AUC=0.7866
- Largest divergences: close range (+1.31%) and counter-attacks (+0.88%) — we predict higher than StatsBomb, likely because they have GK/defender positions
- Stacking our XGBoost + StatsBomb reduces Brier to 0.0769 (AUC 0.8129) — confirms StatsBomb carries orthogonal information

### Key decisions (Phase 10)
- Recommended standalone model: XGBoost full + Platt calibration (Brier=0.0801, AUC=0.7866)
- Recommended model with StatsBomb available: stacked second-stage LR (Brier=0.0769, AUC=0.8129)
- ~6% remaining Brier gap vs StatsBomb is a feature ceiling (freeze-frame), not a modelling ceiling

### In Progress
- Phase 11: Inference Pipeline

### Pending
- Phases 11–12
