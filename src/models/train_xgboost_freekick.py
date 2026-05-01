"""
Free-kick XGBoost Model

Trains a dedicated XGBoost model on direct free-kick shots only.

Production dispatching
-----------------------
In production the inference pipeline dispatches shots to the correct model
based on shot_type:

    shot_type == "Free Kick"  →  this model (fk_xgboost.json + fk_platt_calibrator.joblib)
    otherwise                 →  open-play model (06_xgboost_full.json + 07_platt_xgb_full.joblib)

This separation is necessary because direct free kicks have a fundamentally
different generative process from open-play shots:
  - The kicker has time to plan and place the ball.
  - Defenders form a wall that blocks part of the goal (not captured in our features,
    but the spatial geometry still carries signal).
  - Body position is deliberate — the preferred-foot distinction does not apply.
  - Technique and play-pattern flags have near-zero variance in this population.

Feature set (11 features)
--------------------------
  baseline_spatial (all 6):
    distance_to_goal, visible_angle, distance_to_near_post, distance_to_far_post,
    x_from_goal_line, is_central

  From baseline_categorical (foot only):
    is_right_foot, is_left_foot
    (excluded: is_header, is_other_bodypart — ~100% foot shots, near-zero variance)
    (excluded: all technique flags — minimal variation)
    (excluded: all play-pattern flags — all are "From Free Kick", zero variance)

  From baseline_contextual (situational, minus is_weak_foot and is_first_time):
    minute
    (excluded: is_weak_foot — kickers deliberately choose their foot)
    (excluded: is_first_time — always False in the free-kick population, zero variance)

  advanced_gamestate (all 2):
    score_diff_at_shot, is_late_game

Why the open-play feature set is not reused unchanged:
  Feeding the full 22-feature open-play set to a 4,229-shot dataset would introduce
  near-zero-variance binary flags that act as noise and waste model capacity.
  Pruning to 12 features reduces this noise and improves generalisation on a small
  sample.

Split
-----
Reuses the same match-level split assignments as the open-play model
(04_split_index.parquet) so that if you join predictions across populations,
the same matches appear in the same fold.

Calibration
-----------
XGBoost does not guarantee calibrated probabilities. An inline Platt scaling step
is run on the validation set and the calibrated model is saved. The report compares
raw vs calibrated reliability diagrams so you can judge whether calibration helps.

Outputs
-------
outputs/models/fk_xgboost.json                  raw XGBoost model (native JSON)
outputs/models/fk_platt_calibrator.joblib        Platt scaling LogisticRegression
outputs/tables/fk_val_predictions.parquet        val-set predictions (raw + calibrated)
outputs/tables/fk_test_predictions.parquet       test-set predictions (raw + calibrated)
outputs/tables/fk_metrics.csv                    Brier / log-loss / AUC for all variants
outputs/figures/fk_calibration_curve.png         reliability diagram (raw vs Platt)
outputs/figures/fk_roc_curve.png                 ROC curve
outputs/figures/fk_shap_summary.png             SHAP beeswarm plot
outputs/figures/fk_shap_bar.png                  SHAP mean |value| bar chart
outputs/reports/fk_xgboost_report.md             full report

Run
---
    python -m src.models.train_xgboost_freekick
"""

from __future__ import annotations

import json
import textwrap
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]

PLOT_STYLE = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_cfg(name: str) -> dict:
    with open(ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def _dir(key_path: str) -> Path:
    cfg = _load_cfg("paths")
    node = cfg
    for k in key_path.split("."):
        node = node[k]
    p = ROOT / node
    p.mkdir(parents=True, exist_ok=True)
    return p


def _freekick_features() -> list[str]:
    cfg = _load_cfg("features")
    return cfg["freekick_full"]


# ---------------------------------------------------------------------------
# Data loading and feature engineering
# ---------------------------------------------------------------------------

def load_freekick_data() -> pd.DataFrame:
    """Load Phase 2 free-kick shots and apply the full feature pipeline."""
    from src.features.build import build_features

    interim_dir = ROOT / _load_cfg("paths")["data"]["interim_dir"]
    path = interim_dir / "02_free_kick_shots.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Free-kick sample not found: {path}. "
            "Run `python -m src.data.sample` first."
        )
    df = pd.read_parquet(path)
    df = build_features(df)
    return df


def attach_split_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join match-level split labels from the open-play split index.

    Reusing the same match assignments keeps evaluation consistent: for matches
    that contain both open-play and free-kick shots, both are evaluated on the
    same fold (no cross-split contamination when comparing populations).

    Free-kick matches not present in the split index (edge case: a match had only
    free kicks and no open-play shots) are assigned to 'train' by default.
    """
    tables_dir = ROOT / _load_cfg("paths")["outputs"]["tables_dir"]
    index_path = tables_dir / "04_split_index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Split index not found: {index_path}. "
            "Run `python -m src.data.split` first."
        )
    split_index = (
        pd.read_parquet(index_path)[["match_id", "split"]]
        .drop_duplicates("match_id")
    )
    df = df.merge(split_index, on="match_id", how="left")
    n_missing = df["split"].isna().sum()
    if n_missing > 0:
        print(f"  [WARN] {n_missing} shots have no split label — assigned to 'train'")
        df["split"] = df["split"].fillna("train")
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_model(xgb_cfg: dict) -> XGBClassifier:
    return XGBClassifier(
        objective=xgb_cfg["objective"],
        eval_metric=xgb_cfg["eval_metric"],
        max_depth=xgb_cfg["max_depth"],
        min_child_weight=xgb_cfg["min_child_weight"],
        learning_rate=xgb_cfg["learning_rate"],
        n_estimators=xgb_cfg["n_estimators"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        reg_alpha=xgb_cfg["reg_alpha"],
        reg_lambda=xgb_cfg["reg_lambda"],
        gamma=xgb_cfg["gamma"],
        scale_pos_weight=xgb_cfg["scale_pos_weight"],
        early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
        random_state=xgb_cfg["random_state"],
        n_jobs=xgb_cfg["n_jobs"],
        verbosity=xgb_cfg["verbosity"],
    )


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    xgb_cfg: dict,
) -> tuple[XGBClassifier, np.ndarray, np.ndarray]:
    model = build_model(xgb_cfg)
    model.fit(
        train_df[features].values,
        train_df["is_goal"].values,
        eval_set=[(val_df[features].values, val_df["is_goal"].values)],
        verbose=False,
    )
    p_val  = model.predict_proba(val_df[features].values)[:, 1]
    return model, p_val


# ---------------------------------------------------------------------------
# Platt calibration
# ---------------------------------------------------------------------------

def fit_platt(p_val: np.ndarray, y_val: np.ndarray) -> LogisticRegression:
    """Fit a logistic regression on val-set raw probabilities → calibrated probs."""
    lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    lr.fit(p_val.reshape(-1, 1), y_val)
    return lr


def apply_platt(calibrator: LogisticRegression, p_raw: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(p_raw.reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    split: str,
    variant: str,
) -> dict:
    return {
        "variant": variant,
        "split": split,
        "brier_score": round(brier_score_loss(y_true, p_pred), 6),
        "log_loss": round(log_loss(y_true, p_pred), 6),
        "roc_auc": round(roc_auc_score(y_true, p_pred), 6),
        "n_shots": len(y_true),
        "n_goals": int(y_true.sum()),
        "goal_rate_%": round(y_true.mean() * 100, 2),
        "mean_pred_xg": round(p_pred.mean() * 100, 2),
    }


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def compute_shap(
    model: XGBClassifier,
    X: np.ndarray,
    sample_size: int = 2000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sample = X[idx]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return shap_values, X_sample


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_calibration(
    y_val: np.ndarray,
    prob_dict: dict[str, np.ndarray],
    out_path: Path,
    n_bins: int = 8,
) -> None:
    colors = ["#ff7f0e", "#1f77b4", "#9467bd"]
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        for i, (label, p_pred) in enumerate(prob_dict.items()):
            frac_pos, mean_pred = calibration_curve(
                y_val, p_pred, n_bins=n_bins, strategy="uniform"
            )
            ax.plot(mean_pred, frac_pos, "s-", label=label,
                    color=colors[i % len(colors)], linewidth=1.8, markersize=5)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives (observed goal rate)")
        ax.set_title("Free-kick model — Calibration Curve (val set)")
        ax.set_xlim(0, 0.35)
        ax.set_ylim(0, 0.35)
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def plot_roc(
    y_val: np.ndarray,
    prob_dict: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    colors = ["#ff7f0e", "#1f77b4"]
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        for i, (label, p_pred) in enumerate(prob_dict.items()):
            fpr, tpr, _ = roc_curve(y_val, p_pred)
            auc = roc_auc_score(y_val, p_pred)
            ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})",
                    color=colors[i % len(colors)], linewidth=1.8)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Random")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Free-kick model — ROC Curve (val set)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_names: list[str],
    out_path: Path,
) -> None:
    with plt.rc_context(PLOT_STYLE):
        warnings.filterwarnings("ignore", category=FutureWarning)
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False,
            plot_size=None,
        )
        plt.title("Free-kick model — SHAP Beeswarm (val sample)", fontsize=14)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close("all")


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    out_path: Path,
) -> None:
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap")
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(df["feature"], df["mean_abs_shap"], color="#ff7f0e", edgecolor="white")
        ax.set_xlabel("Mean |SHAP value|  (log-odds units)")
        ax.set_title("Free-kick model — XGBoost Feature Importance via SHAP")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(
    metrics_df: pd.DataFrame,
    features: list[str],
    best_iteration: int,
    platt_coef: float,
    platt_intercept: float,
) -> str:
    p_bar = metrics_df.loc[metrics_df["split"] == "val", "goal_rate_%"].iloc[0] / 100
    null_brier = p_bar * (1 - p_bar)

    lines: list[str] = ["# Free-kick XGBoost Model Report\n"]

    lines.append("## 1. Population & Motivation\n")
    lines.append(textwrap.dedent("""\
        Direct free kicks have a different generative process from open-play shots:

        - The ball is stationary; the kicker has time to plan.
        - Defenders form a wall that partially blocks the view of the goal.
        - The shoot-or-cross decision is predetermined — the xG model only sees the
          shots that were actually taken.
        - Body orientation is deliberate, so the preferred-foot distinction used in
          the open-play model does not carry the same meaning.

        Training a separate model avoids forcing the open-play model to learn a
        joint distribution that conflates two distinct shot populations.  The
        open-play model is trained on ~82k shots; the free-kick model on ~4k shots —
        a thin sample that constrains model complexity.

        **Production dispatching** (see also `src/inference/predict.py`):

        | shot_type       | Model used                              |
        |-----------------|-----------------------------------------|
        | `"Free Kick"`   | `fk_xgboost.json` + Platt calibrator    |
        | anything else   | `06_xgboost_full.json` + Platt from Phase 7 |

    """))

    lines.append("## 2. Feature Set (11 features)\n")
    lines.append("| # | Feature | Group | Rationale |\n|---|---------|-------|-----------|\n")
    groups = {
        "distance_to_goal": "spatial", "visible_angle": "spatial",
        "distance_to_near_post": "spatial", "distance_to_far_post": "spatial",
        "x_from_goal_line": "spatial", "is_central": "spatial",
        "is_right_foot": "categorical", "is_left_foot": "categorical",
        "minute": "contextual",
        "score_diff_at_shot": "gamestate", "is_late_game": "gamestate",
    }
    rationales = {
        "distance_to_goal": "Primary xG driver",
        "visible_angle": "Visible opening toward goal",
        "distance_to_near_post": "Lateral angle signal",
        "distance_to_far_post": "Lateral angle signal",
        "x_from_goal_line": "Depth from goal",
        "is_central": "Central corridor flag",
        "is_right_foot": "Foot side (deliberate choice for FK)",
        "is_left_foot": "Foot side (deliberate choice for FK)",
        "minute": "Match time",
        "score_diff_at_shot": "Game state context",
        "is_late_game": "Late-game pressure flag",
    }
    for i, f in enumerate(features, 1):
        g = groups.get(f, "")
        r = rationales.get(f, "")
        lines.append(f"| {i} | `{f}` | {g} | {r} |")
    lines.append("")

    lines.append("**Excluded from open-play full set (and why):**\n")
    lines.append(textwrap.dedent("""\
        | Excluded feature | Reason |
        |------------------|--------|
        | `is_header`, `is_other_bodypart` | Near-zero variance — direct free kicks are ~100% foot shots |
        | `is_normal_technique`, `is_volley`, `is_diving_header`, `is_exotic_technique` | Minimal technique variation in free-kick population |
        | `is_regular_play`, `is_from_counter`, `is_from_set_piece_restart` | All shots share the same play-pattern — zero variance |
        | `is_weak_foot` | Free-kick takers deliberately choose their foot; the preferred-foot heuristic is not meaningful here |
        | `is_first_time` | Always `False` in the free-kick population — zero variance confirmed in data |
    """))

    lines.append("## 3. Training Configuration\n")
    lines.append(textwrap.dedent(f"""\
        Same `xgboost.yaml` hyperparameters as the open-play model:
        max_depth=4, learning_rate=0.05, min_child_weight=15,
        subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50.

        Best iteration (early stopping on val log loss): **{best_iteration}** trees.

        Note: with only ~4k shots the model is regularisation-constrained.
        Early stopping is critical — without it the model would overfit quickly.
    """))

    lines.append("## 4. Metrics\n")
    lines.append(f"Null Brier score (predict base rate {p_bar:.3f} for every shot): "
                 f"**{null_brier:.6f}**\n")
    lines.append(metrics_df.to_markdown(index=False))
    lines.append("")

    lines.append("## 5. Calibration\n")
    lines.append(textwrap.dedent(f"""\
        **Platt scaling** (logistic regression on val-set raw probabilities):

            p_calibrated = sigmoid({platt_coef:.4f} × logit(p_raw) + {platt_intercept:.4f})

        See `outputs/figures/fk_calibration_curve.png` for the reliability diagram
        (raw XGBoost vs Platt-calibrated vs StatsBomb xG on val set).

        The `fk_platt_calibrator.joblib` artifact is required at inference time —
        raw XGBoost probabilities should never be reported as xG without calibration.
    """))

    lines.append("## 6. SHAP Analysis\n")
    lines.append(textwrap.dedent("""\
        See `outputs/figures/fk_shap_summary.png` (beeswarm) and
        `outputs/figures/fk_shap_bar.png` (mean |SHAP|).

        Expected pattern for free kicks:
        - `distance_to_goal` and `visible_angle` should dominate (geometry drives xG
          even without the defender-wall feature).
        - `is_left_foot` may show a positive effect if left-footed free-kick specialists
          are systematically better at placing the ball in the top corner.
        - `is_first_time` captures rushed attempts (e.g. quick free kicks) — expected
          to show lower xG vs a set free kick from the same location.
    """))

    lines.append("## 7. Limitations\n")
    lines.append(textwrap.dedent("""\
        1. **Defender wall not modelled.** The wall's position and height directly
           reduce the effective visible angle. Without freeze-frame data, the model
           can only learn the average wall penalty from the spatial features.
        2. **Small sample.** ~4k shots limit the model's capacity to learn interactions.
           A deeper tree or more features would overfit.
        3. **No spin / power signal.** Swerving free kicks from 25m and driven low
           free kicks from 20m have very different conversion rates at the same
           location — this difference is invisible without tracking data.
        4. **StatsBomb xG is available for benchmarking.** On the test set, compare
           our fk xG predictions against `statsbomb_xg` to understand the remaining
           gap attributable to missing features.
    """))

    lines.append("## 8. How to Run\n")
    lines.append(
        "    python -m src.models.train_xgboost_freekick\n\n"
        "Output artifacts are in `outputs/models/`, `outputs/tables/`, "
        "`outputs/figures/`, `outputs/reports/`.\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Free-kick XGBoost Model")
    print("=" * 60)

    xgb_cfg  = _load_cfg("xgboost")["xgboost"]
    features = _freekick_features()
    print(f"\nFeature set ({len(features)} features): {features}")

    # --- Load data ---
    print("\nLoading free-kick shots and building features...")
    df = load_freekick_data()
    df = attach_split_labels(df)

    # Verify features present
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # --- Split ---
    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    test_df  = df[df["split"] == "test"].copy()

    print(f"\nTrain : {len(train_df):,} shots | goal rate {train_df['is_goal'].mean()*100:.2f}%")
    print(f"Val   : {len(val_df):,} shots | goal rate {val_df['is_goal'].mean()*100:.2f}%")
    print(f"Test  : {len(test_df):,} shots | goal rate {test_df['is_goal'].mean()*100:.2f}%")

    p_bar      = train_df["is_goal"].mean()
    null_brier = p_bar * (1 - p_bar)
    print(f"\nNull Brier score: {null_brier:.5f}")

    # --- Train XGBoost ---
    print("\nTraining XGBoost...")
    model, p_val_raw = train(train_df, val_df, features, xgb_cfg)
    best_iter = model.best_iteration + 1
    print(f"  Best iteration: {best_iter}  (val log loss = {model.best_score:.5f})")

    # Raw predictions on test set
    p_test_raw = model.predict_proba(test_df[features].values)[:, 1]

    y_val  = val_df["is_goal"].values
    y_test = test_df["is_goal"].values

    print(f"\n  [val ] raw  — Brier={brier_score_loss(y_val,  p_val_raw):.5f}  "
          f"AUC={roc_auc_score(y_val,  p_val_raw):.4f}")
    print(f"  [test] raw  — Brier={brier_score_loss(y_test, p_test_raw):.5f}  "
          f"AUC={roc_auc_score(y_test, p_test_raw):.4f}")

    # --- Platt calibration ---
    print("\nFitting Platt calibrator on val set...")
    calibrator = fit_platt(p_val_raw, y_val)
    p_val_cal  = apply_platt(calibrator, p_val_raw)
    p_test_cal = apply_platt(calibrator, p_test_raw)

    platt_coef      = float(calibrator.coef_[0][0])
    platt_intercept = float(calibrator.intercept_[0])
    print(f"  Platt: coef={platt_coef:.4f}  intercept={platt_intercept:.4f}")
    print(f"  [val ] cal  — Brier={brier_score_loss(y_val,  p_val_cal):.5f}  "
          f"AUC={roc_auc_score(y_val,  p_val_cal):.4f}")
    print(f"  [test] cal  — Brier={brier_score_loss(y_test, p_test_cal):.5f}  "
          f"AUC={roc_auc_score(y_test, p_test_cal):.4f}")

    # --- Metrics ---
    rows = [
        _metrics(y_val,  p_val_raw,  "val",  "xgboost_raw"),
        _metrics(y_test, p_test_raw, "test", "xgboost_raw"),
        _metrics(y_val,  p_val_cal,  "val",  "xgboost_platt"),
        _metrics(y_test, p_test_cal, "test", "xgboost_platt"),
    ]
    if "statsbomb_xg" in val_df.columns:
        p_sb_val  = val_df["statsbomb_xg"].values
        p_sb_test = test_df["statsbomb_xg"].values
        rows += [
            _metrics(y_val,  p_sb_val,  "val",  "statsbomb_xg"),
            _metrics(y_test, p_sb_test, "test", "statsbomb_xg"),
        ]
    metrics_df = pd.DataFrame(rows)

    print("\nMetrics summary:")
    print(metrics_df[["variant", "split", "brier_score", "log_loss", "roc_auc"]].to_string(index=False))

    # --- Save artifacts ---
    models_dir = _dir("outputs.models_dir")
    model.save_model(str(models_dir / "fk_xgboost.json"))
    joblib.dump(calibrator, models_dir / "fk_platt_calibrator.joblib")
    print(f"\nSaved model      : {models_dir.relative_to(ROOT)}/fk_xgboost.json")
    print(f"Saved calibrator : {models_dir.relative_to(ROOT)}/fk_platt_calibrator.joblib")

    tables_dir = _dir("outputs.tables_dir")

    val_preds = val_df[["shot_id", "match_id", "is_goal", "statsbomb_xg"]].copy() \
        if "statsbomb_xg" in val_df.columns \
        else val_df[["shot_id", "match_id", "is_goal"]].copy()
    val_preds["p_fk_xgb_raw"]   = p_val_raw
    val_preds["p_fk_xgb_platt"] = p_val_cal

    test_preds = test_df[["shot_id", "match_id", "is_goal", "statsbomb_xg"]].copy() \
        if "statsbomb_xg" in test_df.columns \
        else test_df[["shot_id", "match_id", "is_goal"]].copy()
    test_preds["p_fk_xgb_raw"]   = p_test_raw
    test_preds["p_fk_xgb_platt"] = p_test_cal

    val_preds.to_parquet(tables_dir  / "fk_val_predictions.parquet",  index=False)
    test_preds.to_parquet(tables_dir / "fk_test_predictions.parquet", index=False)
    metrics_df.to_csv(tables_dir / "fk_metrics.csv", index=False)
    print(f"Saved predictions: {tables_dir.relative_to(ROOT)}/fk_{{val,test}}_predictions.parquet")
    print(f"Saved metrics    : {tables_dir.relative_to(ROOT)}/fk_metrics.csv")

    # --- Plots ---
    figures_dir = _dir("outputs.figures_dir")

    prob_dict_cal: dict[str, np.ndarray] = {
        "XGB raw":   p_val_raw,
        "XGB Platt": p_val_cal,
    }
    if "statsbomb_xg" in val_df.columns:
        prob_dict_cal["StatsBomb xG"] = val_df["statsbomb_xg"].values

    plot_calibration(y_val, prob_dict_cal, figures_dir / "fk_calibration_curve.png")
    print(f"Saved calibration: {figures_dir.relative_to(ROOT)}/fk_calibration_curve.png")

    plot_roc(y_val, {"XGB raw": p_val_raw, "XGB Platt": p_val_cal},
             figures_dir / "fk_roc_curve.png")
    print(f"Saved ROC curve  : {figures_dir.relative_to(ROOT)}/fk_roc_curve.png")

    print("\nComputing SHAP values (val sample)...")
    X_val_arr = val_df[features].values
    shap_values, X_sample = compute_shap(
        model, X_val_arr,
        sample_size=min(2000, len(val_df)),
        random_state=xgb_cfg["random_state"],
    )
    plot_shap_summary(shap_values, X_sample, features,
                      figures_dir / "fk_shap_summary.png")
    print(f"Saved SHAP beeswarm: {figures_dir.relative_to(ROOT)}/fk_shap_summary.png")

    plot_shap_bar(shap_values, features, figures_dir / "fk_shap_bar.png")
    print(f"Saved SHAP bar     : {figures_dir.relative_to(ROOT)}/fk_shap_bar.png")

    # --- Report ---
    reports_dir = _dir("outputs.reports_dir")
    report = generate_report(
        metrics_df,
        features,
        best_iteration=best_iter,
        platt_coef=platt_coef,
        platt_intercept=platt_intercept,
    )
    (reports_dir / "fk_xgboost_report.md").write_text(report, encoding="utf-8")
    print(f"Saved report     : {reports_dir.relative_to(ROOT)}/fk_xgboost_report.md")

    print("\nFree-kick model complete.")
    print("  Production use: load fk_xgboost.json + fk_platt_calibrator.joblib")
    print("  Next: python -m src.inference.predict  (Phase 11)")


if __name__ == "__main__":
    main()
