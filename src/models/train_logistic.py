"""
Phase 5 — Baseline Model: Logistic Regression

Theory
------
Logistic regression models the probability of a goal via the sigmoid of a
linear combination of features:

    p = 1 / (1 + exp(-(b0 + b1*x1 + ... + bn*xn)))

This is the natural baseline for xG for three reasons:
1. It directly outputs calibrated probabilities — no post-hoc calibration needed.
2. Coefficients are interpretable: each is a log-odds ratio.
3. It is a sanity check — if a tree model cannot beat it, the features dominate.

Primary metric: Brier score (MSE of probabilities). We also report log loss and
ROC AUC. Accuracy is intentionally omitted — it is meaningless for imbalanced
probability estimation tasks.

Calibration curve (reliability diagram): bucket predictions into bins and compare
predicted probability to observed goal rate within each bin. A perfectly calibrated
model lies on the diagonal.

Outputs
-------
outputs/models/05_logistic_baseline.joblib      fitted Pipeline (scaler + LR)
outputs/models/05_logistic_full.joblib          fitted Pipeline on full feature set
outputs/tables/05_val_predictions.parquet       val-set predicted probabilities
outputs/tables/05_test_predictions.parquet      test-set predicted probabilities
outputs/tables/05_metrics.csv                   metric table (both feature sets)
outputs/tables/05_coefficients.csv              coefficient table
outputs/figures/05_calibration_curve.png        reliability diagram (val set)
outputs/figures/05_roc_curve.png                ROC curve (val set)
outputs/reports/05_logistic_report.md           full phase report

Run
---
    python -m src.models.train_logistic
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml
from joblib import dump
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
# Identify numeric vs binary features for selective scaling
# ---------------------------------------------------------------------------

BINARY_PREFIXES = ("is_",)


def _is_binary(col: str) -> bool:
    return col.startswith(BINARY_PREFIXES)


def split_feature_types(features: list[str]) -> tuple[list[str], list[str]]:
    """Return (numeric_features, binary_features)."""
    numeric = [f for f in features if not _is_binary(f)]
    binary  = [f for f in features if     _is_binary(f)]
    return numeric, binary


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(random_state: int = 42) -> Pipeline:
    """StandardScaler + L2 Logistic Regression.

    We scale only — the scaler is fitted on training data and applied to
    val/test to prevent data leakage.  L2 regularisation (C=1.0, default)
    provides mild shrinkage appropriate for our feature count (~20 features,
    ~57k training shots).

    solver='lbfgs' is the standard choice for binary logistic regression;
    max_iter=1000 is generous enough to ensure convergence after scaling.
    """
    lr = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
        class_weight=None,  # no reweighting — we want calibrated probabilities
    )
    return Pipeline([("scaler", StandardScaler()), ("lr", lr)])


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    feature_set_name: str,
    random_state: int = 42,
) -> tuple[Pipeline, pd.Series, pd.Series, dict]:
    """Fit pipeline on train, return val and test predicted probabilities plus metrics."""
    X_train = train[features].values
    y_train = train["is_goal"].values

    X_val = val[features].values
    y_val = val["is_goal"].values

    X_test = test[features].values
    y_test = test["is_goal"].values

    pipe = build_pipeline(random_state)
    pipe.fit(X_train, y_train)

    p_val  = pipe.predict_proba(X_val)[:, 1]
    p_test = pipe.predict_proba(X_test)[:, 1]

    def _metrics(y_true: np.ndarray, p_pred: np.ndarray, split: str) -> dict:
        return {
            "feature_set": feature_set_name,
            "split": split,
            "brier_score": round(brier_score_loss(y_true, p_pred), 6),
            "log_loss": round(log_loss(y_true, p_pred), 6),
            "roc_auc": round(roc_auc_score(y_true, p_pred), 6),
            "n_shots": len(y_true),
            "n_goals": int(y_true.sum()),
            "goal_rate_%": round(y_true.mean() * 100, 2),
            "mean_pred_xg": round(p_pred.mean() * 100, 2),
        }

    metrics_rows = [
        _metrics(y_val,  p_val,  "val"),
        _metrics(y_test, p_test, "test"),
    ]

    val_probs  = pd.Series(p_val,  index=val.index,  name=f"p_{feature_set_name}")
    test_probs = pd.Series(p_test, index=test.index, name=f"p_{feature_set_name}")

    return pipe, val_probs, test_probs, metrics_rows


# ---------------------------------------------------------------------------
# Coefficient extraction
# ---------------------------------------------------------------------------

def extract_coefficients(pipe: Pipeline, features: list[str]) -> pd.DataFrame:
    lr = pipe.named_steps["lr"]
    coefs = lr.coef_[0]
    intercept = lr.intercept_[0]

    rows = [{"feature": f, "coefficient": c, "odds_ratio": np.exp(c)}
            for f, c in zip(features, coefs)]
    rows.append({"feature": "(intercept)", "coefficient": intercept, "odds_ratio": np.exp(intercept)})

    df = pd.DataFrame(rows).sort_values("coefficient", key=abs, ascending=False)
    df["coefficient"] = df["coefficient"].round(4)
    df["odds_ratio"]  = df["odds_ratio"].round(4)
    return df.reset_index(drop=True)


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


def plot_calibration(
    y_val: np.ndarray,
    prob_dict: dict[str, np.ndarray],
    out_path: Path,
    n_bins: int = 10,
) -> None:
    """Reliability diagram comparing multiple models/feature sets."""
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for (label, p_pred), color in zip(prob_dict.items(), colors):
            fraction_pos, mean_pred = calibration_curve(y_val, p_pred, n_bins=n_bins, strategy="uniform")
            ax.plot(mean_pred, fraction_pos, "s-", label=label, color=color, linewidth=1.8, markersize=5)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration", alpha=0.6)

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives (observed goal rate)")
        ax.set_title("Phase 5 — Calibration Curve (val set)")
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
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))

        colors = ["#1f77b4", "#ff7f0e"]
        for (label, p_pred), color in zip(prob_dict.items(), colors):
            fpr, tpr, _ = roc_curve(y_val, p_pred)
            auc = roc_auc_score(y_val, p_pred)
            ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", color=color, linewidth=1.8)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Random")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Phase 5 — ROC Curve (val set)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def plot_coefficient_bar(coef_df: pd.DataFrame, out_path: Path, feature_set_name: str) -> None:
    """Horizontal bar chart of logistic regression coefficients (top N by |coef|)."""
    top = coef_df[coef_df["feature"] != "(intercept)"].head(20).copy()
    top = top.sort_values("coefficient")

    colors = ["#d62728" if c > 0 else "#1f77b4" for c in top["coefficient"]]

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(top["feature"], top["coefficient"], color=colors, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Coefficient (log-odds units)")
        ax.set_title(f"Phase 5 — Logistic Regression Coefficients\n({feature_set_name} feature set)")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    metrics_df: pd.DataFrame,
    coef_df: pd.DataFrame,
    features_baseline: list[str],
    features_full: list[str],
) -> str:
    null_brier = metrics_df.loc[metrics_df["split"] == "val", "goal_rate_%"].iloc[0] / 100
    null_brier_val = null_brier * (1 - null_brier)

    lines: list[str] = ["# Phase 5 — Baseline Model: Logistic Regression\n"]

    lines.append("## 1. Theory\n")
    lines.append(textwrap.dedent("""\
        ### The logistic function

        Logistic regression maps a linear combination of features to [0, 1] via the sigmoid:

            p = 1 / (1 + exp(-(b0 + b1*x1 + ... + bn*xn)))

        This is the natural model for xG because:
        - The output is natively a probability (no post-hoc clamping needed).
        - It is *natively calibrated* under correct model specification — predicted
          probabilities reflect actual goal rates in the training population.
        - Coefficients are interpretable: exp(b_i) is the odds ratio for a one-unit
          increase in feature x_i, holding others constant.

        ### Loss function: log loss (cross-entropy)

            L = -(1/n) * Σ [ y_i log(p_i) + (1 - y_i) log(1 - p_i) ]

        Log loss penalises confident wrong predictions heavily — a model that says 99%
        when the true label is 0 incurs a much larger penalty than one that says 60%.
        Accuracy, by contrast, treats all errors equally regardless of confidence.

        ### Brier score (primary metric)

            BS = (1/n) * Σ (p_i - y_i)²

        This is the MSE of predicted probabilities. For xG we sum probabilities to get
        expected goals per player / team / game — miscalibration compounds, so probability
        quality matters more than discrimination alone.

        The **null Brier score** (predicting the base rate for every shot) is
        `p̄ × (1 - p̄)`. A good model must beat this.

        ### Calibration curve (reliability diagram)

        Predictions are binned and the observed goal rate is compared to mean predicted
        probability in each bin. A perfectly calibrated model lies on the diagonal y = x.

        ### Preprocessing

        Logistic regression is sensitive to feature scale — if `distance_to_goal` ranges
        0–80 while `is_header` is binary (0/1), the gradient update for distance will
        dominate.  We apply `StandardScaler` (mean=0, std=1) to all features on the
        *training set only* and apply the fitted scaler to val/test.  Binary indicator
        features become ±σ after scaling — this is harmless for L2-regularised logistic
        regression.

    """))

    lines.append("## 2. Feature Sets\n")
    lines.append(f"- **Baseline** ({len(features_baseline)} features): "
                 + ", ".join(f"`{f}`" for f in features_baseline) + "\n")
    lines.append(f"- **Full** ({len(features_full)} features): adds "
                 + ", ".join(f"`{f}`" for f in features_full if f not in features_baseline) + "\n")

    lines.append("## 3. Metrics\n")
    lines.append(f"Null Brier score (predict base rate for all shots): **{null_brier_val:.6f}**\n")
    lines.append(metrics_df.to_markdown(index=False))
    lines.append("")
    lines.append(textwrap.dedent("""\
        **How to read:**
        - Lower Brier score = better. Beat the null model (above).
        - Lower log loss = better.
        - Higher ROC AUC = better discrimination. AUC = 0.5 is random, 1.0 is perfect.
        - `mean_pred_xg` should be close to `goal_rate_%` — otherwise the model is
          systematically over/under-predicting.
    """))

    lines.append("## 4. Coefficients (baseline feature set, top features by |coef|)\n")
    lines.append(coef_df.head(15).to_markdown(index=False))
    lines.append(textwrap.dedent("""\

        **Interpretation guide:**
        - A coefficient of −0.08 for `distance_to_goal` means each extra yard from goal
          multiplies the *odds* of scoring by exp(−0.08) ≈ 0.92 — about an 8% odds
          reduction per yard.
        - `odds_ratio < 1` → feature reduces probability (e.g. greater distance).
        - `odds_ratio > 1` → feature increases probability (e.g. is_header with a
          positive coefficient in certain contexts).
    """))

    lines.append("## 5. Key Findings\n")
    val_baseline = metrics_df[(metrics_df["feature_set"] == "baseline") & (metrics_df["split"] == "val")]
    val_full     = metrics_df[(metrics_df["feature_set"] == "full")     & (metrics_df["split"] == "val")]
    if len(val_baseline) and len(val_full):
        bs_base = val_baseline["brier_score"].iloc[0]
        bs_full = val_full["brier_score"].iloc[0]
        auc_base = val_baseline["roc_auc"].iloc[0]
        auc_full = val_full["roc_auc"].iloc[0]
        lines.append(textwrap.dedent(f"""\
            - Null Brier score: {null_brier_val:.6f}
            - Baseline model Brier score (val): {bs_base:.6f} — beats null by {null_brier_val - bs_base:.6f}
            - Full model Brier score (val): {bs_full:.6f}
            - Adding game-state features changed AUC from {auc_base:.4f} → {auc_full:.4f}
            - See reliability diagram and coefficient chart in `outputs/figures/`.
        """))

    lines.append("## 6. Next Steps\n")
    lines.append(textwrap.dedent("""\
        Phase 6 will train an XGBoost model on the same feature sets.  Key questions:
        - Can XGBoost improve Brier score and ROC AUC over logistic regression?
        - Does it preserve calibration, or does it need post-hoc recalibration (Phase 7)?
        - Which features does XGBoost consider most important (SHAP analysis)?
    """))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 5 — Baseline Model: Logistic Regression")
    print("=" * 60)

    cfg_paths = _load_cfg("paths")
    random_state = 42

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

    # Train baseline model
    print("\n--- Baseline feature set ---")
    pipe_baseline, val_p_base, test_p_base, metrics_base = train_and_evaluate(
        train, val, test, features_baseline, "baseline", random_state
    )
    for m in metrics_base:
        print(f"  [{m['split']:5s}] Brier={m['brier_score']:.5f}  LogLoss={m['log_loss']:.5f}  AUC={m['roc_auc']:.4f}")

    # Train full model
    print("\n--- Full feature set ---")
    pipe_full, val_p_full, test_p_full, metrics_full = train_and_evaluate(
        train, val, test, features_full, "full", random_state
    )
    for m in metrics_full:
        print(f"  [{m['split']:5s}] Brier={m['brier_score']:.5f}  LogLoss={m['log_loss']:.5f}  AUC={m['roc_auc']:.4f}")

    # Null model Brier score
    p_bar = train["is_goal"].mean()
    null_brier = p_bar * (1 - p_bar)
    print(f"\nNull Brier score (predict {p_bar*100:.2f}% for all shots): {null_brier:.5f}")

    # Save models
    models_dir = _dir("outputs.models_dir")
    dump(pipe_baseline, models_dir / "05_logistic_baseline.joblib")
    dump(pipe_full,     models_dir / "05_logistic_full.joblib")
    print(f"\nSaved models to {models_dir.relative_to(ROOT)}/")

    # Save predictions
    tables_dir = _dir("outputs.tables_dir")

    val_preds = val[["shot_id", "match_id", "is_goal", "statsbomb_xg"]].copy()
    val_preds["p_baseline"] = val_p_base.values
    val_preds["p_full"]     = val_p_full.values
    val_preds.to_parquet(tables_dir / "05_val_predictions.parquet", index=False)

    test_preds = test[["shot_id", "match_id", "is_goal", "statsbomb_xg"]].copy()
    test_preds["p_baseline"] = test_p_base.values
    test_preds["p_full"]     = test_p_full.values
    test_preds.to_parquet(tables_dir / "05_test_predictions.parquet", index=False)

    print(f"Saved predictions to {tables_dir.relative_to(ROOT)}/")

    # Save metrics
    metrics_df = pd.DataFrame(metrics_base + metrics_full)
    metrics_df.to_csv(tables_dir / "05_metrics.csv", index=False)
    print("\nMetrics summary:")
    print(metrics_df[["feature_set", "split", "brier_score", "log_loss", "roc_auc"]].to_string(index=False))

    # Save coefficients
    coef_df = extract_coefficients(pipe_baseline, features_baseline)
    coef_df.to_csv(tables_dir / "05_coefficients.csv", index=False)
    print("\nTop 10 coefficients (baseline model):")
    print(coef_df.head(10)[["feature", "coefficient", "odds_ratio"]].to_string(index=False))

    # Plots
    figures_dir = _dir("outputs.figures_dir")
    y_val = val["is_goal"].values

    plot_calibration(
        y_val,
        {"Baseline (20 feat)": val_p_base.values, "Full (22 feat)": val_p_full.values},
        figures_dir / "05_calibration_curve.png",
    )
    print(f"\nSaved calibration curve : {figures_dir.relative_to(ROOT)}/05_calibration_curve.png")

    plot_roc(
        y_val,
        {"Baseline": val_p_base.values, "Full": val_p_full.values},
        figures_dir / "05_roc_curve.png",
    )
    print(f"Saved ROC curve         : {figures_dir.relative_to(ROOT)}/05_roc_curve.png")

    plot_coefficient_bar(coef_df, figures_dir / "05_coefficients.png", "baseline")
    print(f"Saved coefficient chart : {figures_dir.relative_to(ROOT)}/05_coefficients.png")

    # Report
    reports_dir = _dir("outputs.reports_dir")
    report = generate_report(metrics_df, coef_df, features_baseline, features_full)
    (reports_dir / "05_logistic_report.md").write_text(report, encoding="utf-8")
    print(f"Saved report            : {reports_dir.relative_to(ROOT)}/05_logistic_report.md")

    print("\nPhase 5 complete.")
    print("  Next: python -m src.models.train_xgboost")


if __name__ == "__main__":
    main()
