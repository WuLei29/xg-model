"""
Phase 6 — Boosted Tree Model: XGBoost

Theory
------
Gradient boosting builds an ensemble of shallow trees sequentially.  At step t
a new tree h_t is fitted to the negative gradient of the loss with respect to
the current ensemble's predictions F_{t-1}:

    F_t(x) = F_{t-1}(x) + η · h_t(x)

For binary log loss the negative gradient at example i is (y_i − p_i) — the
raw residual between label and predicted probability.  The ensemble therefore
iteratively reduces its own errors, converging to the minimum of the loss.

Why XGBoost for xG?
-------------------
- Captures non-linear feature interactions automatically (e.g. distance × angle
  interaction is worth more than either feature alone at certain ranges).
- Handles missing values natively — no imputation needed.
- `binary:logistic` objective outputs probabilities in [0, 1], comparable to
  logistic regression output.

Calibration caveat
------------------
Unlike logistic regression, XGBoost does NOT guarantee calibrated probabilities.
The model minimises log loss, which pushes predictions toward the correct label,
but the resulting probability scale can be compressed at the extremes.  This is
why Phase 7 (calibration) always follows Phase 6.

Key hyperparameters
-------------------
max_depth          : tree depth — controls the order of feature interactions.
                     Depth 1 = stumps (no interactions); depth 4 = up to 4-way
                     interactions.  Deeper trees overfit more easily.
learning_rate (η)  : shrinks each tree's contribution.  Smaller η → more trees
                     are needed to fit the data but each step is more conservative,
                     reducing variance.  Standard range: 0.01–0.1.
n_estimators       : maximum number of trees; early stopping picks the actual best.
min_child_weight   : minimum sum of sample weights in a leaf.  Prevents the model
                     from creating leaves based on very few rare-event examples.
subsample          : fraction of training rows sampled per tree (row bagging).
                     Adds variance reduction similar to random forests.
colsample_bytree   : fraction of features sampled per tree.  Adds feature
                     diversity and reduces co-adaptation across trees.
reg_alpha / lambda : L1 / L2 regularisation on leaf weights.
scale_pos_weight   : ratio of negative to positive class used to reweight the
                     gradient.  Set to 1 to preserve calibrated probabilities —
                     reweighting helps recall but distorts probability estimates.

Early stopping
--------------
We monitor validation log loss and stop when it has not improved for
`early_stopping_rounds` consecutive rounds.  This is the most powerful
regularisation available: the model stops exactly when it starts to overfit the
training set.  The final `best_iteration` is used for all subsequent predictions.

SHAP values
-----------
SHAP (SHapley Additive exPlanations) assigns each feature an additive
contribution to each individual prediction.  For tree models this is computed
exactly (not sampled) via Lundberg's TreeExplainer algorithm.

Key properties:
- sum of SHAP values + expected_value = raw predicted log-odds for that shot.
- Global importance = mean |SHAP value| across all shots in the dataset.
- Unlike gain or split-count importance, SHAP accounts for feature interactions
  and correlated features — it is the most reliable importance measure.

Common pitfall: high AUC does not mean well-calibrated probabilities.  Always
check the reliability diagram alongside AUC.

Outputs
-------
outputs/models/06_xgboost_baseline.json       XGBoost native model (baseline)
outputs/models/06_xgboost_full.json           XGBoost native model (full)
outputs/tables/06_val_predictions.parquet     val-set: LR + XGB predictions
outputs/tables/06_test_predictions.parquet    test-set: LR + XGB predictions
outputs/tables/06_metrics.csv                 all models, both splits, both phases
outputs/figures/06_calibration_curve.png      XGBoost vs Logistic (val set)
outputs/figures/06_roc_curve.png              XGBoost vs Logistic (val set)
outputs/figures/06_shap_summary.png           SHAP beeswarm plot (baseline)
outputs/figures/06_shap_bar.png               SHAP mean |value| bar (baseline)
outputs/reports/06_xgboost_report.md          full phase report

Run
---
    python -m src.models.train_xgboost
"""

from __future__ import annotations

import json
import textwrap
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]

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


def _load_feature_list(feature_set: str) -> list[str]:
    path = ROOT / "outputs" / "tables" / "03_feature_list.json"
    with open(path) as f:
        data = json.load(f)
    return data["sets"][feature_set]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed = ROOT / _load_cfg("paths")["data"]["processed_dir"]
    train = pd.read_parquet(processed / "04_train.parquet")
    val   = pd.read_parquet(processed / "04_val.parquet")
    test  = pd.read_parquet(processed / "04_test.parquet")
    return train, val, test


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(xgb_cfg: dict) -> XGBClassifier:
    """Construct XGBClassifier from config dict.

    XGBoost is scale-invariant so no StandardScaler is needed — unlike logistic
    regression, the gradient updates are based on tree splits, not feature
    magnitudes.
    """
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


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def _metrics(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    split: str,
    feature_set: str,
    model: str,
) -> dict:
    return {
        "model": model,
        "feature_set": feature_set,
        "split": split,
        "brier_score": round(brier_score_loss(y_true, p_pred), 6),
        "log_loss": round(log_loss(y_true, p_pred), 6),
        "roc_auc": round(roc_auc_score(y_true, p_pred), 6),
        "n_shots": len(y_true),
        "n_goals": int(y_true.sum()),
        "goal_rate_%": round(y_true.mean() * 100, 2),
        "mean_pred_xg": round(p_pred.mean() * 100, 2),
    }


def train_and_evaluate(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    feature_set_name: str,
    xgb_cfg: dict,
) -> tuple[XGBClassifier, pd.Series, pd.Series, list[dict]]:
    """Fit XGBClassifier with early stopping; return probabilities and metrics."""
    X_train = train[features].values
    y_train = train["is_goal"].values
    X_val   = val[features].values
    y_val   = val["is_goal"].values
    X_test  = test[features].values
    y_test  = test["is_goal"].values

    model = build_model(xgb_cfg)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    n_trees = model.best_iteration + 1
    print(f"    Early stopping: best iteration = {n_trees} "
          f"(val log loss = {model.best_score:.5f})")

    p_val  = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    metrics_rows = [
        _metrics(y_val,  p_val,  "val",  feature_set_name, "xgboost"),
        _metrics(y_test, p_test, "test", feature_set_name, "xgboost"),
    ]

    val_probs  = pd.Series(p_val,  index=val.index,  name=f"p_xgb_{feature_set_name}")
    test_probs = pd.Series(p_test, index=test.index, name=f"p_xgb_{feature_set_name}")

    return model, val_probs, test_probs, metrics_rows


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------

def compute_shap(
    model: XGBClassifier,
    X: np.ndarray,
    feature_names: list[str],
    sample_size: int = 5000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (shap_values, X_sample) for a random subset of rows.

    We sample to keep computation fast.  For ~82k shots the full SHAP
    computation takes ~30 seconds; 5000 rows gives a reliable estimate of
    global importance in a few seconds.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sample = X[idx]

    explainer  = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return shap_values, X_sample


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PLOT_STYLE = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
}

_COLORS = {
    "lr_baseline":  "#aec7e8",
    "lr_full":      "#1f77b4",
    "xgb_baseline": "#ffbb78",
    "xgb_full":     "#ff7f0e",
}


def plot_calibration(
    y_val: np.ndarray,
    prob_dict: dict[str, np.ndarray],
    out_path: Path,
    n_bins: int = 10,
) -> None:
    """Reliability diagram for multiple models."""
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))

        color_list = list(_COLORS.values())
        for i, (label, p_pred) in enumerate(prob_dict.items()):
            frac_pos, mean_pred = calibration_curve(
                y_val, p_pred, n_bins=n_bins, strategy="uniform"
            )
            color = color_list[i % len(color_list)]
            ax.plot(mean_pred, frac_pos, "s-", label=label, color=color,
                    linewidth=1.8, markersize=5)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect", alpha=0.6)

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives (observed goal rate)")
        ax.set_title("Phase 6 — Calibration Curve (val set)")
        ax.set_xlim(0, 0.55)
        ax.set_ylim(0, 0.55)
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
    color_list = list(_COLORS.values())
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))

        for i, (label, p_pred) in enumerate(prob_dict.items()):
            fpr, tpr, _ = roc_curve(y_val, p_pred)
            auc = roc_auc_score(y_val, p_pred)
            color = color_list[i % len(color_list)]
            ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", color=color, linewidth=1.8)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Random")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Phase 6 — ROC Curve (val set)")
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
    """Beeswarm plot: each dot is one shot; colour = feature value; x-axis = SHAP."""
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(9, 7))
        # shap.summary_plot creates its own figure; we capture it by passing show=False
        # and then saving the current figure.
        warnings.filterwarnings("ignore", category=FutureWarning)
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_names,
            show=False,
            plot_size=None,
        )
        plt.title("Phase 6 — SHAP Beeswarm (baseline, val sample)", fontsize=14)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close("all")


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    out_path: Path,
) -> None:
    """Horizontal bar chart of mean |SHAP| value per feature."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap")

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(df["feature"], df["mean_abs_shap"], color="#ff7f0e", edgecolor="white")
        ax.set_xlabel("Mean |SHAP value|  (log-odds units)")
        ax.set_title("Phase 6 — XGBoost Feature Importance via SHAP\n(baseline feature set)")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    metrics_df: pd.DataFrame,
    phase5_df: pd.DataFrame | None,
    features_baseline: list[str],
    features_full: list[str],
    best_iter_baseline: int,
    best_iter_full: int,
) -> str:
    p_bar = metrics_df.loc[metrics_df["split"] == "val", "goal_rate_%"].iloc[0] / 100
    null_brier = p_bar * (1 - p_bar)

    lines: list[str] = ["# Phase 6 — Boosted Tree Model: XGBoost\n"]

    lines.append("## 1. Theory\n")
    lines.append(textwrap.dedent("""\
        ### Gradient boosting

        Gradient boosting builds an additive ensemble of shallow decision trees.
        At each iteration t, a new tree h_t is fitted to minimise:

            L(F_{t-1}(x) + η·h_t(x))

        where L is the log loss and η is the learning rate (shrinkage).  For
        binary log loss the negative gradient at example i is simply (y_i − p_i),
        the residual between the true label and the current predicted probability.
        The ensemble therefore keeps "correcting" its errors across iterations.

        ### Why XGBoost for xG?

        Logistic regression models the log-odds as a *linear* function of features.
        But football shot quality is not linear:
        - The relationship between distance and goal probability is not linear —
          it follows the geometry of the visible goal angle.
        - A header from 6 yards differs from a header from 20 yards in a way that
          a purely additive model cannot fully express.
        - First-time shots from a cross behave differently from first-time shots
          from a through ball, even at the same distance — an interaction effect.

        XGBoost captures these non-linearities and interactions automatically by
        splitting features at data-driven thresholds and combining them in trees.

        ### Calibration caveat

        Unlike logistic regression, XGBoost does NOT guarantee calibrated
        probabilities.  The model minimises log loss, but the resulting probability
        scale can be compressed at the extremes.  Common pattern: the model is
        over-confident at low probabilities (predicts 5% but true rate is 3%) and
        under-confident at high probabilities (predicts 40% but true rate is 50%).
        This is why Phase 7 (Platt scaling / isotonic regression) follows.

        ### Early stopping

        We set `n_estimators=2000` as an upper bound and monitor validation log
        loss.  Training halts when the loss has not improved for 50 consecutive
        rounds.  This is the primary regularisation mechanism — the model stops
        exactly when it starts to overfit the training set.

        ### SHAP values

        SHAP (SHapley Additive exPlanations) computes, for each prediction, the
        contribution of each feature.  For tree models this is computed exactly
        (not sampled) via the TreeExplainer algorithm.

            prediction_i = E[output] + Σ_j SHAP_ij

        Unlike gain importance (which counts splits) or cover importance (which
        counts observations), SHAP correctly attributes credit in the presence
        of correlated features and feature interactions.

    """))

    lines.append("## 2. Feature Sets\n")
    lines.append(f"- **Baseline** ({len(features_baseline)} features): "
                 + ", ".join(f"`{f}`" for f in features_baseline) + "\n")
    extra = [f for f in features_full if f not in features_baseline]
    lines.append(f"- **Full** ({len(features_full)} features): adds "
                 + ", ".join(f"`{f}`" for f in extra) + "\n")
    lines.append(f"- Baseline model best iteration: **{best_iter_baseline}** trees\n")
    lines.append(f"- Full model best iteration: **{best_iter_full}** trees\n")

    lines.append("## 3. Metrics\n")
    lines.append(f"Null Brier score (predict base rate for all shots): **{null_brier:.6f}**\n")

    xgb_metrics = metrics_df[metrics_df["model"] == "xgboost"]
    lines.append(xgb_metrics.to_markdown(index=False))
    lines.append("")

    if phase5_df is not None:
        lines.append("### Comparison with Phase 5 (Logistic Regression)\n")
        combined = pd.concat([phase5_df, xgb_metrics], ignore_index=True)
        combined = combined.sort_values(["split", "model", "feature_set"])
        lines.append(combined[["model", "feature_set", "split",
                                "brier_score", "log_loss", "roc_auc"]].to_markdown(index=False))
        lines.append("")

    lines.append(textwrap.dedent("""\
        **How to read:**
        - Lower Brier score and log loss = better probability quality.
        - Higher ROC AUC = better discrimination (ranking ability).
        - `mean_pred_xg` vs `goal_rate_%`: close = well-calibrated at global level.
        - A high AUC with worse Brier score vs logistic regression is a common XGBoost
          pattern — it ranks shots well but probabilities are slightly miscalibrated.
          Phase 7 corrects this.
    """))

    lines.append("## 4. SHAP Analysis\n")
    lines.append(textwrap.dedent("""\
        See `outputs/figures/06_shap_summary.png` (beeswarm) and
        `outputs/figures/06_shap_bar.png` (mean |SHAP|).

        **Interpreting the beeswarm plot:**
        - Each row is a feature; each dot is one shot.
        - Dot position on x-axis = SHAP value (positive → increases predicted xG).
        - Dot colour = feature value (red = high, blue = low).
        - Expected pattern: high distance → negative SHAP (reduces xG); high visible
          angle → positive SHAP (increases xG).

        **SHAP vs logistic regression coefficients:**
        - Logistic regression gives one number per feature (the coefficient).
        - SHAP gives a distribution of contributions — the effect of a feature can
          vary across shots (e.g. `minute` may have near-zero SHAP for most shots
          but large SHAP in the 85th minute when a team is losing).
    """))

    lines.append("## 5. Key Findings\n")
    xgb_val = xgb_metrics[(xgb_metrics["split"] == "val") &
                           (xgb_metrics["feature_set"] == "baseline")]
    if len(xgb_val):
        bs = xgb_val["brier_score"].iloc[0]
        auc = xgb_val["roc_auc"].iloc[0]
        lines.append(f"- XGBoost baseline Brier score (val): **{bs:.6f}** "
                     f"(beats null by {null_brier - bs:.6f})\n")
        lines.append(f"- XGBoost baseline ROC AUC (val): **{auc:.4f}**\n")
    if phase5_df is not None:
        lr_val = phase5_df[(phase5_df["split"] == "val") &
                           (phase5_df["feature_set"] == "baseline")]
        if len(lr_val):
            lr_bs  = lr_val["brier_score"].iloc[0]
            lr_auc = lr_val["roc_auc"].iloc[0]
            if len(xgb_val):
                delta_bs  = lr_bs  - xgb_val["brier_score"].iloc[0]
                delta_auc = xgb_val["roc_auc"].iloc[0] - lr_auc
                lines.append(f"- XGBoost improved Brier score by **{delta_bs:+.6f}** "
                              f"and AUC by **{delta_auc:+.4f}** over logistic regression "
                              f"(baseline features).\n")

    lines.append("## 6. Next Steps\n")
    lines.append(textwrap.dedent("""\
        Phase 7 will calibrate both the logistic regression and XGBoost models.
        Expected findings:
        - Logistic regression is already well-calibrated (coefficients directly
          estimate log-odds, sigmoid maps to probabilities).
        - XGBoost may benefit from Platt scaling — a thin logistic layer fitted
          on a held-out calibration set that rescales the model's probability outputs.
        - We will compare Brier score before and after calibration with reliability
          diagrams to quantify the improvement.
    """))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 6 — Boosted Tree Model: XGBoost")
    print("=" * 60)

    xgb_cfg      = _load_cfg("xgboost")["xgboost"]
    random_state = xgb_cfg["random_state"]

    # Load splits
    train, val, test = load_splits()
    print(f"\nTrain : {len(train):,} shots | goal rate {train['is_goal'].mean()*100:.2f}%")
    print(f"Val   : {len(val):,} shots | goal rate {val['is_goal'].mean()*100:.2f}%")
    print(f"Test  : {len(test):,} shots | goal rate {test['is_goal'].mean()*100:.2f}%")

    # Feature sets
    features_baseline = _load_feature_list("baseline")
    features_full     = _load_feature_list("full")
    print(f"\nBaseline features : {len(features_baseline)}")
    print(f"Full features     : {len(features_full)}")

    # Null Brier score
    p_bar      = train["is_goal"].mean()
    null_brier = p_bar * (1 - p_bar)
    print(f"\nNull Brier score  : {null_brier:.5f}")

    # Train baseline model
    print("\n--- Baseline feature set ---")
    model_baseline, val_p_base, test_p_base, metrics_base = train_and_evaluate(
        train, val, test, features_baseline, "baseline", xgb_cfg
    )
    for m in metrics_base:
        print(f"  [{m['split']:5s}] Brier={m['brier_score']:.5f}  "
              f"LogLoss={m['log_loss']:.5f}  AUC={m['roc_auc']:.4f}")

    # Train full model
    print("\n--- Full feature set ---")
    model_full, val_p_full, test_p_full, metrics_full = train_and_evaluate(
        train, val, test, features_full, "full", xgb_cfg
    )
    for m in metrics_full:
        print(f"  [{m['split']:5s}] Brier={m['brier_score']:.5f}  "
              f"LogLoss={m['log_loss']:.5f}  AUC={m['roc_auc']:.4f}")

    # Save XGBoost models (native JSON format — portable, human-readable)
    models_dir = _dir("outputs.models_dir")
    model_baseline.save_model(str(models_dir / "06_xgboost_baseline.json"))
    model_full.save_model(str(models_dir / "06_xgboost_full.json"))
    print(f"\nSaved models to {models_dir.relative_to(ROOT)}/")

    # Save predictions (include Phase 5 LR columns if available)
    tables_dir = _dir("outputs.tables_dir")

    p5_val_path  = tables_dir / "05_val_predictions.parquet"
    p5_test_path = tables_dir / "05_test_predictions.parquet"

    if p5_val_path.exists():
        val_preds  = pd.read_parquet(p5_val_path)
        test_preds = pd.read_parquet(p5_test_path)
    else:
        val_preds  = val[["shot_id", "match_id", "is_goal", "statsbomb_xg"]].copy()
        test_preds = test[["shot_id", "match_id", "is_goal", "statsbomb_xg"]].copy()

    val_preds["p_xgb_baseline"]  = val_p_base.values
    val_preds["p_xgb_full"]      = val_p_full.values
    test_preds["p_xgb_baseline"] = test_p_base.values
    test_preds["p_xgb_full"]     = test_p_full.values

    val_preds.to_parquet(tables_dir  / "06_val_predictions.parquet",  index=False)
    test_preds.to_parquet(tables_dir / "06_test_predictions.parquet", index=False)
    print(f"Saved predictions to {tables_dir.relative_to(ROOT)}/")

    # Metrics table — load Phase 5 CSV for comparison
    xgb_metrics_df = pd.DataFrame(metrics_base + metrics_full)
    xgb_metrics_df.to_csv(tables_dir / "06_xgb_metrics.csv", index=False)

    phase5_df: pd.DataFrame | None = None
    p5_metrics_path = tables_dir / "05_metrics.csv"
    if p5_metrics_path.exists():
        phase5_df = pd.read_csv(p5_metrics_path)
        phase5_df.insert(0, "model", "logistic")
        combined_metrics = pd.concat([phase5_df, xgb_metrics_df], ignore_index=True)
    else:
        combined_metrics = xgb_metrics_df

    combined_metrics.to_csv(tables_dir / "06_metrics.csv", index=False)
    print("\nMetrics summary (XGBoost):")
    print(xgb_metrics_df[["feature_set", "split", "brier_score",
                           "log_loss", "roc_auc"]].to_string(index=False))

    # Plots
    figures_dir = _dir("outputs.figures_dir")
    y_val = val["is_goal"].values

    # Build prob_dict for comparison plots
    prob_dict: dict[str, np.ndarray] = {}
    if p5_val_path.exists():
        lr_preds = pd.read_parquet(p5_val_path)
        if "p_baseline" in lr_preds.columns:
            prob_dict["LR baseline"] = lr_preds["p_baseline"].values
        if "p_full" in lr_preds.columns:
            prob_dict["LR full"]     = lr_preds["p_full"].values
    prob_dict["XGB baseline"] = val_p_base.values
    prob_dict["XGB full"]     = val_p_full.values

    plot_calibration(
        y_val, prob_dict,
        figures_dir / "06_calibration_curve.png",
    )
    print(f"\nSaved calibration curve : {figures_dir.relative_to(ROOT)}/06_calibration_curve.png")

    plot_roc(
        y_val, prob_dict,
        figures_dir / "06_roc_curve.png",
    )
    print(f"Saved ROC curve         : {figures_dir.relative_to(ROOT)}/06_roc_curve.png")

    # SHAP analysis on baseline model
    print("\nComputing SHAP values (baseline model, val sample)...")
    X_val_arr = val[features_baseline].values
    shap_values, X_sample = compute_shap(
        model_baseline, X_val_arr, features_baseline,
        sample_size=5000, random_state=random_state,
    )

    plot_shap_summary(shap_values, X_sample, features_baseline,
                      figures_dir / "06_shap_summary.png")
    print(f"Saved SHAP beeswarm     : {figures_dir.relative_to(ROOT)}/06_shap_summary.png")

    plot_shap_bar(shap_values, features_baseline,
                  figures_dir / "06_shap_bar.png")
    print(f"Saved SHAP bar chart    : {figures_dir.relative_to(ROOT)}/06_shap_bar.png")

    # Report
    reports_dir = _dir("outputs.reports_dir")
    report = generate_report(
        xgb_metrics_df,
        phase5_df,
        features_baseline,
        features_full,
        best_iter_baseline=model_baseline.best_iteration + 1,
        best_iter_full=model_full.best_iteration + 1,
    )
    (reports_dir / "06_xgboost_report.md").write_text(report, encoding="utf-8")
    print(f"Saved report            : {reports_dir.relative_to(ROOT)}/06_xgboost_report.md")

    print("\nPhase 6 complete.")
    print("  Next: python -m src.evaluation.calibration  (Phase 7)")


if __name__ == "__main__":
    main()
