"""
Phase 7 — Calibration

Theory
------
Calibration asks a single question: when the model says a shot has a 15% chance
of being a goal, do approximately 15% of those shots actually go in?

A model can have excellent discrimination (rank shots by difficulty perfectly)
and be completely miscalibrated at the same time. For xG specifically,
calibration matters more than discrimination because we *sum* predicted
probabilities. A team taking 10 shots predicted at 0.12 each should total 1.2 xG.
If the model systematically predicts 0.12 when the true rate is 0.08, every
player and team comparison is biased.

Reliability diagram (calibration curve)
----------------------------------------
Predicted probabilities are sorted and binned (e.g. 10 uniform bins). Within each
bin, we compare:
  - Mean predicted probability (x-axis)
  - Observed goal rate (y-axis)
A perfectly calibrated model lies on the diagonal. Points above the diagonal mean
the model is under-confident (predicts 10% but 15% of those shots go in). Points
below mean over-confident.

Brier score decomposition
--------------------------
The Brier score BS = (1/n) Σ (p_i − y_i)² decomposes into three terms:

    BS = Calibration + Resolution − Uncertainty

- Calibration: how far bin-mean predictions are from observed rates (lower = better)
- Resolution: how much the bin rates vary from the overall base rate (higher = better;
  a model that just predicts p̄ everywhere has zero resolution)
- Uncertainty: irreducible noise = p̄(1 − p̄)

Improving calibration reduces the Calibration term without affecting Resolution.
Post-hoc recalibration cannot improve AUC (it is a monotonic transformation of
scores), but it directly reduces the Calibration component of Brier score.

Post-hoc calibration methods
------------------------------
Platt scaling
  Fit a logistic regression on the raw model outputs (probabilities or log-odds)
  using the calibration set:
      p_cal = sigmoid(a * logit(p_raw) + b)
  Learns a linear recalibration in log-odds space. Works well when the model is
  approximately calibrated but has a systematic scale or shift error.
  Needs only ~1000 samples to fit reliably.

Isotonic regression
  Fit a piecewise-constant monotone function mapping raw probabilities to
  observed rates. Fully non-parametric — can correct any monotone miscalibration
  pattern. Requires more data (~10,000+ samples) to avoid overfitting; on smaller
  calibration sets it memorises the data and degrades on test.

When to use which:
  Platt scaling is the safe default for most situations.
  Isotonic regression is useful when the miscalibration is severe and non-linear
  AND the calibration set is large. With ~12k val shots and ~10% goal rate we have
  ~1,250 positive examples — Platt scaling is appropriate; isotonic is marginal.

Calibration-discrimination tradeoff
-------------------------------------
Post-hoc calibration is a monotone transformation of scores, so ROC AUC is
unchanged by definition. However, Platt scaling can occasionally slightly
reduce AUC if the logistic layer is regularised too heavily. We use near-zero
regularisation (C=1e10) to make it a pure scale/shift correction.

Setup: fit calibrators on validation set, evaluate on test set
---------------------------------------------------------------
XGBoost was trained with early stopping monitored on the val set, so the val
set is "seen" data. However, early stopping only used the log-loss value, not
the probability scale — the model never optimised for Platt's (a, b) parameters.
Using val for Platt fitting is therefore acceptable and standard practice.

Outputs
-------
outputs/models/07_platt_lr_baseline.joblib
outputs/models/07_platt_lr_full.joblib
outputs/models/07_platt_xgb_baseline.joblib
outputs/models/07_platt_xgb_full.joblib
outputs/models/07_isotonic_xgb_baseline.joblib
outputs/models/07_isotonic_xgb_full.joblib
outputs/tables/07_val_predictions.parquet   raw + calibrated probabilities (val)
outputs/tables/07_test_predictions.parquet  raw + calibrated probabilities (test)
outputs/tables/07_metrics.csv               Brier/AUC before and after calibration
outputs/figures/07_reliability_grid.png     2×2 reliability diagrams (before/after)
outputs/figures/07_brier_comparison.png     grouped bar chart (raw vs Platt)
outputs/reports/07_calibration_report.md

Run
---
    python -m src.evaluation.calibration
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml
from joblib import dump
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import FunctionTransformer

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


# ---------------------------------------------------------------------------
# Calibration fitters
# ---------------------------------------------------------------------------

EPS = 1e-7  # clip probabilities away from 0/1 before logit


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(p_cal: np.ndarray, y_cal: np.ndarray) -> LogisticRegression:
    """Fit Platt scaling: logistic regression on logit(p_raw).

    C=1e10 → near-zero regularisation so the fit is a pure scale/shift
    correction without shrinking the coefficients toward zero.
    """
    X = _logit(p_cal).reshape(-1, 1)
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(X, y_cal)
    return lr


def apply_platt(platt: LogisticRegression, p_raw: np.ndarray) -> np.ndarray:
    X = _logit(p_raw).reshape(-1, 1)
    return platt.predict_proba(X)[:, 1]


def fit_isotonic(p_cal: np.ndarray, y_cal: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal)
    return iso


def apply_isotonic(iso: IsotonicRegression, p_raw: np.ndarray) -> np.ndarray:
    return iso.predict(p_raw)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    model: str,
    feature_set: str,
    calibration: str,
    split: str,
) -> dict:
    return {
        "model": model,
        "feature_set": feature_set,
        "calibration": calibration,
        "split": split,
        "brier_score": round(brier_score_loss(y_true, p_pred), 6),
        "log_loss": round(log_loss(y_true, p_pred), 6),
        "roc_auc": round(roc_auc_score(y_true, p_pred), 6),
        "mean_pred_xg": round(p_pred.mean() * 100, 2),
        "goal_rate_%": round(y_true.mean() * 100, 2),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PLOT_STYLE = {
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
}

_FMT = mticker.PercentFormatter(xmax=1, decimals=0)


def _reliability_ax(
    ax: plt.Axes,
    y_true: np.ndarray,
    curves: dict[str, np.ndarray],
    title: str,
    n_bins: int = 10,
) -> None:
    """Draw reliability curves on a single Axes."""
    styles = [
        ("raw",   "#aaaaaa", "o--", 1.5),
        ("platt", "#ff7f0e", "s-",  2.0),
    ]
    for (label, p_pred), (_, color, ls, lw) in zip(curves.items(), styles):
        frac, mean_pred = calibration_curve(y_true, p_pred, n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred, frac, ls, color=color, label=label, linewidth=lw, markersize=5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="perfect")
    ax.set_title(title)
    ax.set_xlim(0, 0.55)
    ax.set_ylim(0, 0.55)
    ax.xaxis.set_major_formatter(_FMT)
    ax.yaxis.set_major_formatter(_FMT)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)


def plot_reliability_grid(
    y_val: np.ndarray,
    panels: list[tuple[str, dict[str, np.ndarray]]],
    out_path: Path,
) -> None:
    """2×2 grid of reliability diagrams — one per model/feature-set combo."""
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
        axes_flat = axes.flatten()

        for ax, (title, curves) in zip(axes_flat, panels):
            _reliability_ax(ax, y_val, curves, title)

        fig.supxlabel("Mean predicted probability", y=0.02)
        fig.supylabel("Fraction of positives (observed goal rate)", x=0.02)
        fig.suptitle("Phase 7 — Reliability Diagrams: Raw vs Platt Scaling (val set)", fontsize=13)
        fig.tight_layout(rect=[0.03, 0.03, 1, 0.97])
        fig.savefig(out_path)
        plt.close(fig)


def plot_brier_comparison(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: raw vs Platt Brier score for each model × feature-set."""
    val_df = metrics_df[
        (metrics_df["split"] == "test") &
        (metrics_df["calibration"].isin(["raw", "platt"]))
    ].copy()

    val_df["label"] = val_df["model"] + "\n" + val_df["feature_set"]
    pivot = val_df.pivot(index="label", columns="calibration", values="brier_score")

    x = np.arange(len(pivot))
    w = 0.35

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))

        bars_raw   = ax.bar(x - w/2, pivot["raw"],   w, label="Raw",   color="#aaaaaa")
        bars_platt = ax.bar(x + w/2, pivot["platt"], w, label="Platt", color="#ff7f0e")

        # Annotate improvement
        for xpos, (raw_val, platt_val) in zip(x, zip(pivot["raw"], pivot["platt"])):
            delta = raw_val - platt_val
            if abs(delta) > 0:
                ax.annotate(
                    f"−{delta*1000:.2f}×10⁻³",
                    xy=(xpos, max(raw_val, platt_val) + 0.0002),
                    ha="center", va="bottom", fontsize=8, color="#444444",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index)
        ax.set_ylabel("Brier score (lower = better)")
        ax.set_title("Phase 7 — Brier Score: Raw vs Platt Scaling (test set)")
        ax.legend()
        ax.set_ylim(0.078, ax.get_ylim()[1] * 1.04)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(metrics_df: pd.DataFrame, null_brier: float) -> str:
    lines: list[str] = ["# Phase 7 — Calibration\n"]

    lines.append("## 1. Theory\n")
    lines.append(textwrap.dedent("""\
        ### What calibration means

        A model is calibrated if its predicted probabilities match observed frequencies.
        Formally, for any predicted value p:

            P(Y = 1 | model predicts p) ≈ p

        When we say the model outputs 0.15 for a shot, we want exactly 15% of those
        shots to be goals. For xG this is critical because we sum probabilities over a
        match to get "expected goals". A model that outputs 0.08 when the true rate is
        0.12 will systematically undercount xG for every player and team.

        ### Reliability diagram

        Predictions are binned (e.g. 10 uniform bins). Within each bin we plot
        observed goal rate (y) against mean predicted probability (x). The diagonal
        y = x is perfect calibration. The most common XGBoost pattern is:

        - **Low probability region (0–5%)**: slight over-prediction (model says 3%,
          true rate is 2%) — the model is too confident on near-impossible shots.
        - **High probability region (30%+)**: slight under-prediction (model says 35%,
          true rate is 42%) — the model is under-confident on clear-cut chances.

        This "compressed" pattern is typical of gradient boosting models trained with
        log loss. It does not impair ranking ability (AUC is unaffected) but it
        distorts the probability scale used for xG summation.

        ### Brier score decomposition

        BS = Calibration + Resolution − Uncertainty

        - **Calibration term**: mean squared gap between bin-predicted probability and
          bin-observed rate. Post-hoc calibration directly reduces this term.
        - **Resolution term**: variance of bin-observed rates around the base rate.
          A model that always predicts p̄ has zero resolution. Better features → higher
          resolution. Calibration cannot improve resolution.
        - **Uncertainty term**: p̄(1 − p̄), the irreducible noise. Constant for a
          given dataset.

        This means calibration cannot push Brier score below the "floor" set by the
        model's resolution. What we gain is that the probability scale becomes
        trustworthy for summation.

        ### Platt scaling

        Fit a logistic regression on logit(p_raw) using a held-out calibration set:

            p_cal = sigmoid(a · logit(p_raw) + b)

        The coefficient a corrects scale (compresses or expands the probability range)
        and b corrects systematic bias (shift). With C=1e10 (near-zero regularisation)
        this is a pure two-parameter fit, not a new model.

        ### Isotonic regression

        A fully non-parametric monotone mapping from raw probabilities to observed
        rates. More flexible than Platt, but needs more data to avoid overfitting. With
        ~12,500 val shots and ~10% goal rate (~1,250 positives), isotonic regression
        risks over-fitting the calibration set; Platt scaling is the safer choice here.

        ### Why AUC does not change

        Platt scaling is a strictly monotone transformation of the raw scores (it
        preserves rank order). ROC AUC depends only on rank order, so it is unchanged
        by definition. A post-hoc calibrator can never improve or hurt discrimination.

    """))

    lines.append("## 2. Results\n")
    p_bar = metrics_df["goal_rate_%"].iloc[0] / 100
    lines.append(f"Null Brier score: **{null_brier:.6f}**\n")

    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    lines.append(test_df[["model", "feature_set", "calibration",
                           "brier_score", "log_loss", "roc_auc"]].to_markdown(index=False))
    lines.append("")

    lines.append("## 3. Analysis\n")

    for fs in ["baseline", "full"]:
        for mdl in ["logistic", "xgboost"]:
            raw_row   = test_df[(test_df["model"] == mdl) &
                                (test_df["feature_set"] == fs) &
                                (test_df["calibration"] == "raw")]
            platt_row = test_df[(test_df["model"] == mdl) &
                                (test_df["feature_set"] == fs) &
                                (test_df["calibration"] == "platt")]
            if len(raw_row) and len(platt_row):
                delta = raw_row["brier_score"].iloc[0] - platt_row["brier_score"].iloc[0]
                lines.append(
                    f"- **{mdl} / {fs}**: Platt scaling "
                    + (f"improved Brier by {delta*1000:.3f}×10⁻³"
                       if delta > 0 else f"had no improvement (delta={delta*1000:.3f}×10⁻³)")
                    + f" | AUC unchanged at {platt_row['roc_auc'].iloc[0]:.4f}\n"
                )

    lines.append("\n## 4. Recommendation\n")
    lines.append(textwrap.dedent("""\
        For all downstream phases (subgroup analysis, inference, reporting):

        - **XGBoost models**: use **Platt-calibrated** probabilities. The raw XGBoost
          probabilities are slightly miscalibrated; Platt scaling corrects the scale
          at negligible cost and makes the xG values trustworthy for summation.
        - **Logistic regression**: the raw probabilities are already well-calibrated
          (the sigmoid maps log-odds to probabilities correctly). Platt scaling makes
          a negligible difference and the raw probabilities can be used directly.
        - **Do not use isotonic regression** at this dataset size — with ~1,250
          positive examples in the calibration set, it risks overfitting.

        The saved calibrators (`07_platt_*.joblib`) will be loaded by the inference
        pipeline in Phase 11.
    """))

    lines.append("## 5. Next Steps\n")
    lines.append(textwrap.dedent("""\
        Phase 8 — Subgroup Analysis: evaluate the calibrated XGBoost model on
        specific shot types (headers, long-range, first-time, counterattack, etc.)
        to identify where the model systematically over- or under-predicts.
    """))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 7 — Calibration")
    print("=" * 60)

    tables_dir  = _dir("outputs.tables_dir")
    models_dir  = _dir("outputs.models_dir")
    figures_dir = _dir("outputs.figures_dir")
    reports_dir = _dir("outputs.reports_dir")

    # Load val and test predictions (all four models in one file from Phase 6)
    val_preds  = pd.read_parquet(tables_dir / "06_val_predictions.parquet")
    test_preds = pd.read_parquet(tables_dir / "06_test_predictions.parquet")

    y_val  = val_preds["is_goal"].values
    y_test = test_preds["is_goal"].values

    p_bar      = y_val.mean()
    null_brier = p_bar * (1 - p_bar)
    print(f"\nNull Brier score : {null_brier:.5f}")
    print(f"Val  set size    : {len(y_val):,} shots  |  goal rate {p_bar*100:.2f}%")
    print(f"Test set size    : {len(y_test):,} shots  |  goal rate {y_test.mean()*100:.2f}%")

    # Column mapping: (model_name, feature_set, val_col, test_col)
    model_cols = [
        ("logistic", "baseline", "p_baseline",     "p_baseline"),
        ("logistic", "full",     "p_full",          "p_full"),
        ("xgboost",  "baseline", "p_xgb_baseline",  "p_xgb_baseline"),
        ("xgboost",  "full",     "p_xgb_full",      "p_xgb_full"),
    ]

    all_metrics: list[dict] = []

    # Containers for output prediction columns
    val_out  = val_preds[["shot_id", "match_id", "is_goal", "statsbomb_xg"]].copy()
    test_out = test_preds[["shot_id", "match_id", "is_goal", "statsbomb_xg"]].copy()

    # Reliability diagram panels
    panels: list[tuple[str, dict[str, np.ndarray]]] = []

    for model_name, feature_set, vcol, tcol in model_cols:
        p_val_raw  = val_preds[vcol].values
        p_test_raw = test_preds[tcol].values

        tag = f"{model_name}_{feature_set}"
        print(f"\n--- {model_name} / {feature_set} ---")

        # Raw metrics
        for y, p, split in [(y_val, p_val_raw, "val"), (y_test, p_test_raw, "test")]:
            all_metrics.append(compute_metrics(y, p, model_name, feature_set, "raw", split))
        bs_raw = brier_score_loss(y_test, p_test_raw)
        print(f"  Raw  Brier (test): {bs_raw:.5f}")

        # Copy raw cols to output
        val_out[f"p_{tag}"]  = p_val_raw
        test_out[f"p_{tag}"] = p_test_raw

        # Platt scaling — fit on val, evaluate on test
        platt = fit_platt(p_val_raw, y_val)
        p_val_platt  = apply_platt(platt, p_val_raw)
        p_test_platt = apply_platt(platt, p_test_raw)

        for y, p, split in [(y_val, p_val_platt, "val"), (y_test, p_test_platt, "test")]:
            all_metrics.append(compute_metrics(y, p, model_name, feature_set, "platt", split))

        bs_platt = brier_score_loss(y_test, p_test_platt)
        delta    = bs_raw - bs_platt
        print(f"  Platt Brier (test): {bs_platt:.5f}  (delta = {delta:+.6f})")

        dump(platt, models_dir / f"07_platt_{tag}.joblib")

        val_out[f"p_{tag}_platt"]  = p_val_platt
        test_out[f"p_{tag}_platt"] = p_test_platt

        # Isotonic regression — fit on val, evaluate on test (informational only)
        iso = fit_isotonic(p_val_raw, y_val)
        p_test_iso = apply_isotonic(iso, p_test_raw)
        p_val_iso  = apply_isotonic(iso, p_val_raw)

        for y, p, split in [(y_val, p_val_iso, "val"), (y_test, p_test_iso, "test")]:
            all_metrics.append(compute_metrics(y, p, model_name, feature_set, "isotonic", split))

        bs_iso = brier_score_loss(y_test, p_test_iso)
        print(f"  Isotonic Brier (test): {bs_iso:.5f}")

        dump(iso, models_dir / f"07_isotonic_{tag}.joblib")

        # Reliability panel (raw vs platt on val)
        panels.append((
            f"{model_name} / {feature_set}",
            {"raw": p_val_raw, "platt": p_val_platt},
        ))

    # Save predictions
    val_out.to_parquet(tables_dir  / "07_val_predictions.parquet",  index=False)
    test_out.to_parquet(tables_dir / "07_test_predictions.parquet", index=False)
    print(f"\nSaved calibrated predictions to {tables_dir.relative_to(ROOT)}/")

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(tables_dir / "07_metrics.csv", index=False)

    # Plots
    plot_reliability_grid(y_val, panels, figures_dir / "07_reliability_grid.png")
    print(f"Saved reliability grid  : outputs/figures/07_reliability_grid.png")

    plot_brier_comparison(metrics_df, figures_dir / "07_brier_comparison.png")
    print(f"Saved Brier comparison  : outputs/figures/07_brier_comparison.png")

    # Report
    report = generate_report(metrics_df, null_brier)
    (reports_dir / "07_calibration_report.md").write_text(report, encoding="utf-8")
    print(f"Saved report            : outputs/reports/07_calibration_report.md")

    # Summary table
    print("\nTest-set summary (raw vs Platt):")
    summary = metrics_df[
        (metrics_df["split"] == "test") &
        (metrics_df["calibration"].isin(["raw", "platt"]))
    ][["model", "feature_set", "calibration", "brier_score", "log_loss", "roc_auc"]]
    print(summary.sort_values(["model", "feature_set", "calibration"]).to_string(index=False))

    print("\nPhase 7 complete.")
    print("  Next: python -m src.evaluation.subgroups  (Phase 8)")


if __name__ == "__main__":
    main()
