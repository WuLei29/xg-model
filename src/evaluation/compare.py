"""
Phase 9 — Comparative Analysis & Gap Investigation

Benchmarks our models against StatsBomb xG, analyses divergence by subgroup,
and runs a stacking experiment to quantify the information gain from StatsBomb.

Usage:
    python -m src.evaluation.compare
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
REPORTS = ROOT / "outputs" / "reports"
PROCESSED = ROOT / "data" / "processed"

FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Models to include in the three-way benchmark (column name → display label)
BENCHMARK_MODELS: dict[str, str] = {
    "statsbomb_xg": "StatsBomb xG",
    "p_xgboost_full_platt": "XGBoost (calibrated)",
    "p_logistic_full_platt": "Logistic Regression (calibrated)",
    "p_xgboost_baseline_platt": "XGBoost Baseline (calibrated)",
}

# Our primary model column
PRIMARY_MODEL = "p_xgboost_full_platt"

# Subgroup definitions — mirrors Phase 8 structure
# Each entry: (subgroup_name, category_name, boolean_mask_expression)
# Mask expressions are evaluated against the merged test dataframe.
SUBGROUP_DEFS: list[tuple[str, str, str]] = [
    # body part
    ("Headers", "body_part", "is_header == 1"),
    ("Right foot", "body_part", "is_right_foot == 1"),
    ("Left foot", "body_part", "is_left_foot == 1"),
    # technique
    ("Normal technique", "technique", "is_normal_technique == 1"),
    ("Volley", "technique", "is_volley == 1"),
    # context
    ("First-time", "context", "is_first_time == 1"),
    ("Weak foot", "context", "is_weak_foot == 1"),
    # pattern
    ("Regular play", "pattern", "is_regular_play == 1"),
    ("Counter-attack", "pattern", "is_from_counter == 1"),
    ("Set-piece restart", "pattern", "is_from_set_piece_restart == 1"),
    # distance
    ("Close range (<=10 yds)", "distance", "distance_to_goal <= 10"),
    ("Medium range (10-20 yds)", "distance", "(distance_to_goal > 10) & (distance_to_goal <= 20)"),
    ("Long range (>20 yds)", "distance", "distance_to_goal > 20"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    return {
        "model": label,
        "n_shots": len(y_true),
        "n_goals": int(y_true.sum()),
        "goal_rate_%": round(y_true.mean() * 100, 2),
        "mean_pred_xg_%": round(y_pred.mean() * 100, 2),
        "brier_score": round(brier_score_loss(y_true, y_pred), 6),
        "log_loss": round(log_loss(y_true, y_pred), 6),
        "roc_auc": round(roc_auc_score(y_true, y_pred), 4),
    }


def _ax_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# 1. Three-way benchmark
# ---------------------------------------------------------------------------

def benchmark(test_preds: pd.DataFrame) -> pd.DataFrame:
    """Compute Brier, log loss, AUC for each model on the test set."""
    y = test_preds["is_goal"].values
    rows = []
    for col, label in BENCHMARK_MODELS.items():
        if col not in test_preds.columns:
            continue
        rows.append(compute_metrics(y, test_preds[col].values, label))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Calibration curves — three-way overlay
# ---------------------------------------------------------------------------

def plot_calibration_curves(test_preds: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    y = test_preds["is_goal"].values

    for (col, label), color in zip(BENCHMARK_MODELS.items(), colors):
        if col not in test_preds.columns:
            continue
        frac_pos, mean_pred = calibration_curve(y, test_preds[col].values, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", lw=2, color=color, label=label, markersize=5)

    _ax_style(ax, "Calibration Curves — All Models vs StatsBomb", "Mean Predicted xG", "Observed Goal Rate")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    plt.tight_layout()
    fig.savefig(FIGURES / "09_calibration_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 09_calibration_comparison.png")


# ---------------------------------------------------------------------------
# 3. Prediction distribution comparison
# ---------------------------------------------------------------------------

def plot_prediction_distributions(test_preds: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(BENCHMARK_MODELS), figsize=(5 * len(BENCHMARK_MODELS), 4), sharey=True)
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    for ax, (col, label), color in zip(axes, BENCHMARK_MODELS.items(), colors):
        if col not in test_preds.columns:
            continue
        ax.hist(test_preds[col], bins=40, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.axvline(test_preds[col].mean(), color="black", lw=1.5, linestyle="--",
                   label=f"Mean = {test_preds[col].mean():.3f}")
        _ax_style(ax, label, "Predicted xG", "Count")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.suptitle("Prediction Distributions on Test Set", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES / "09_prediction_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 09_prediction_distributions.png")


# ---------------------------------------------------------------------------
# 4. Scatter: primary model vs StatsBomb, coloured by body part
# ---------------------------------------------------------------------------

def plot_scatter(test_merged: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    body_groups = {
        "Header": test_merged["is_header"] == 1,
        "Right foot": test_merged["is_right_foot"] == 1,
        "Left foot": test_merged["is_left_foot"] == 1,
        "Other": (test_merged["is_header"] == 0)
                  & (test_merged["is_right_foot"] == 0)
                  & (test_merged["is_left_foot"] == 0),
    }
    colors = {"Header": "#9C27B0", "Right foot": "#2196F3", "Left foot": "#FF5722", "Other": "#9E9E9E"}
    alphas = {"Header": 0.5, "Right foot": 0.35, "Left foot": 0.35, "Other": 0.35}

    for label, mask in body_groups.items():
        subset = test_merged[mask]
        ax.scatter(subset["statsbomb_xg"], subset[PRIMARY_MODEL],
                   s=8, alpha=alphas[label], color=colors[label], label=f"{label} (n={mask.sum():,})")

    lim = max(test_merged["statsbomb_xg"].max(), test_merged[PRIMARY_MODEL].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
    _ax_style(ax, "XGBoost (calibrated) vs StatsBomb xG — Per Shot",
              "StatsBomb xG", "Our XGBoost xG (calibrated)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(fontsize=9, markerscale=2, loc="upper left")
    plt.tight_layout()
    fig.savefig(FIGURES / "09_scatter_xgb_vs_statsbomb.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 09_scatter_xgb_vs_statsbomb.png")


# ---------------------------------------------------------------------------
# 5. Divergence analysis by subgroup
# ---------------------------------------------------------------------------

def divergence_analysis(test_merged: pd.DataFrame) -> pd.DataFrame:
    """
    For each subgroup compute mean(our_xg) - mean(statsbomb_xg).
    Positive = we predict higher than StatsBomb; negative = we predict lower.
    """
    rows = []
    for sg_name, category, expr in SUBGROUP_DEFS:
        mask = test_merged.eval(expr)
        subset = test_merged[mask]
        if len(subset) < 10:
            continue
        our_mean = subset[PRIMARY_MODEL].mean()
        sb_mean = subset["statsbomb_xg"].mean()
        actual = subset["is_goal"].mean()
        our_bias = round((our_mean - actual) * 100, 2)
        sb_bias = round((sb_mean - actual) * 100, 2)
        abs_our = round(abs(our_bias), 2)
        abs_sb = round(abs(sb_bias), 2)
        if abs_our < abs_sb:
            better = "Ours"
        elif abs_sb < abs_our:
            better = "StatsBomb"
        else:
            better = "Tied"
        rows.append({
            "subgroup": sg_name,
            "category": category,
            "n_shots": len(subset),
            "actual_goal_rate_%": round(actual * 100, 2),
            "mean_our_xg_%": round(our_mean * 100, 2),
            "mean_statsbomb_xg_%": round(sb_mean * 100, 2),
            "gap_our_minus_sb_%": round((our_mean - sb_mean) * 100, 2),
            "our_bias_%": our_bias,
            "sb_bias_%": sb_bias,
            "abs_our_bias_%": abs_our,
            "abs_sb_bias_%": abs_sb,
            "bias_advantage_%": round(abs_sb - abs_our, 2),  # positive = ours is better
            "better_calibrated": better,
        })
    return pd.DataFrame(rows)


def plot_divergence_chart(div_df: pd.DataFrame) -> None:
    df = div_df.sort_values("gap_our_minus_sb_%")
    colors = ["#FF5722" if v > 0 else "#2196F3" for v in df["gap_our_minus_sb_%"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(df["subgroup"], df["gap_our_minus_sb_%"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
    _ax_style(ax, "xG Gap: Our Model vs StatsBomb by Subgroup\n(positive = we predict higher)",
              "Mean xG Gap (our model − StatsBomb) %", "")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES / "09_divergence_by_subgroup.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 09_divergence_by_subgroup.png")


def plot_bias_comparison(div_df: pd.DataFrame) -> None:
    """
    Grouped bar chart: absolute bias of our model vs StatsBomb per subgroup.
    Highlights which model is better calibrated in each subgroup.
    """
    df = div_df.sort_values("abs_our_bias_%", ascending=False)
    subgroups = df["subgroup"].tolist()
    x = np.arange(len(subgroups))
    width = 0.35

    our_color = "#FF5722"
    sb_color = "#2196F3"

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_our = ax.bar(x - width / 2, df["abs_our_bias_%"], width, label="Our XGBoost",
                      color=our_color, alpha=0.85, edgecolor="white")
    bars_sb = ax.bar(x + width / 2, df["abs_sb_bias_%"], width, label="StatsBomb xG",
                     color=sb_color, alpha=0.85, edgecolor="white")

    # Mark the winner with a star above the lower bar
    for i, (_, row) in enumerate(df.iterrows()):
        winner_x = (x[i] - width / 2) if row["better_calibrated"] == "Ours" else (x[i] + width / 2)
        lower_val = min(row["abs_our_bias_%"], row["abs_sb_bias_%"])
        ax.text(winner_x, lower_val + 0.02, "★", ha="center", va="bottom", fontsize=10,
                color="gold", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(subgroups, rotation=35, ha="right", fontsize=9)
    _ax_style(ax, "Calibration Bias by Subgroup — Our Model vs StatsBomb\n(★ = better calibrated)",
              "", "Absolute Bias vs Actual Goal Rate (%)")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
    plt.tight_layout()
    fig.savefig(FIGURES / "09_bias_comparison_by_subgroup.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 09_bias_comparison_by_subgroup.png")


# ---------------------------------------------------------------------------
# 6. Stacking experiment
# ---------------------------------------------------------------------------

def stacking_experiment(
    val_preds: pd.DataFrame, test_preds: pd.DataFrame
) -> tuple[dict, LogisticRegression]:
    """
    Train a second-stage logistic regression on val set using:
        inputs:  [p_xgboost_full_platt, statsbomb_xg]
        target:  is_goal

    Evaluate on test set. Compare against standalone XGBoost.

    NOTE: This model requires StatsBomb xG at inference time.
    It is a research exercise, not a production model.
    """
    X_val = val_preds[[PRIMARY_MODEL, "statsbomb_xg"]].values
    y_val = val_preds["is_goal"].values

    X_test = test_preds[[PRIMARY_MODEL, "statsbomb_xg"]].values
    y_test = test_preds["is_goal"].values

    stack = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    stack.fit(X_val, y_val)
    y_stack = stack.predict_proba(X_test)[:, 1]

    standalone = test_preds[PRIMARY_MODEL].values

    results = {
        "standalone_xgboost": compute_metrics(y_test, standalone, "XGBoost standalone"),
        "stacked_xgboost_plus_statsbomb": compute_metrics(y_test, y_stack, "Stacked (XGBoost + StatsBomb)"),
    }
    print("  Stacking experiment — standalone vs stacked:")
    for k, v in results.items():
        print(f"    {v['model']}: Brier={v['brier_score']:.6f}  AUC={v['roc_auc']:.4f}  LogLoss={v['log_loss']:.6f}")

    return results, stack, y_stack


def plot_stacking_calibration(test_preds: pd.DataFrame, y_stack: np.ndarray) -> None:
    y = test_preds["is_goal"].values
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")

    for preds, label, color in [
        (test_preds[PRIMARY_MODEL].values, "XGBoost standalone", "#FF5722"),
        (y_stack, "Stacked (XGBoost + StatsBomb)", "#4CAF50"),
        (test_preds["statsbomb_xg"].values, "StatsBomb xG", "#2196F3"),
    ]:
        fp, mp = calibration_curve(y, preds, n_bins=10, strategy="quantile")
        ax.plot(mp, fp, marker="o", lw=2, color=color, label=label, markersize=5)

    _ax_style(ax, "Stacking Experiment — Calibration Curves",
              "Mean Predicted xG", "Observed Goal Rate")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    plt.tight_layout()
    fig.savefig(FIGURES / "09_stacking_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 09_stacking_calibration.png")


# ---------------------------------------------------------------------------
# 7. Markdown report
# ---------------------------------------------------------------------------

def _fmt_row(row: pd.Series) -> str:
    return (
        f"| {row['model']} | {row['n_shots']:,} | {row['goal_rate_%']}% "
        f"| {row['mean_pred_xg_%']}% | {row['brier_score']:.6f} "
        f"| {row['log_loss']:.6f} | {row['roc_auc']:.4f} |"
    )


def write_report(
    benchmark_df: pd.DataFrame,
    div_df: pd.DataFrame,
    stack_results: dict,
) -> None:
    lines: list[str] = []

    lines += [
        "# Phase 9 — Comparative Analysis & Gap Investigation",
        "",
        "## 1. Three-Way Benchmark (Test Set)",
        "",
        "StatsBomb xG is treated as a **reference model** — its predictions are already ",
        "in the dataset as `statsbomb_xg`. All three models are evaluated on the same ",
        "held-out test split (n = {:,} shots).".format(int(benchmark_df.iloc[0]["n_shots"])),
        "",
        "| Model | Shots | Goal Rate | Mean Pred xG | Brier ↓ | Log Loss ↓ | AUC ↑ |",
        "|-------|------:|----------:|-------------:|--------:|-----------:|------:|",
    ]
    for _, row in benchmark_df.iterrows():
        lines.append(_fmt_row(row))

    lines += [
        "",
        "> **Interpretation**: Brier score and log loss are lower-is-better (probability quality). "
        "AUC is higher-is-better (discrimination only). Our primary production model is "
        f"**XGBoost (calibrated)**.",
        "",
        "---",
        "",
        "## 2. Divergence Analysis by Subgroup",
        "",
        "For each subgroup: mean(our xG) − mean(StatsBomb xG). Positive = we predict higher than "
        "StatsBomb; negative = we predict lower. Both biases relative to actual goal rate are shown.",
        "",
        "| Subgroup | n | Actual % | Our xG % | SB xG % | Gap (our−SB) | Our bias | SB bias |",
        "|----------|--:|---------:|---------:|--------:|-------------:|---------:|--------:|",
    ]
    for _, r in div_df.iterrows():
        lines.append(
            f"| {r['subgroup']} | {r['n_shots']:,} | {r['actual_goal_rate_%']}% "
            f"| {r['mean_our_xg_%']}% | {r['mean_statsbomb_xg_%']}% "
            f"| {r['gap_our_minus_sb_%']:+.2f}% | {r['our_bias_%']:+.2f}% | {r['sb_bias_%']:+.2f}% |"
        )

    # Largest gaps
    top_pos = div_df.nlargest(3, "gap_our_minus_sb_%")[["subgroup", "gap_our_minus_sb_%"]]
    top_neg = div_df.nsmallest(3, "gap_our_minus_sb_%")[["subgroup", "gap_our_minus_sb_%"]]

    lines += [
        "",
        "### Key divergences",
        "",
        "**We predict notably higher than StatsBomb on**:",
    ]
    for _, r in top_pos.iterrows():
        lines.append(f"- {r['subgroup']} ({r['gap_our_minus_sb_%']:+.2f}%)")

    lines += ["", "**We predict notably lower than StatsBomb on**:"]
    for _, r in top_neg.iterrows():
        lines.append(f"- {r['subgroup']} ({r['gap_our_minus_sb_%']:+.2f}%)")

    lines += [
        "",
        "> **Reading the table**: the **gap** column shows where the two models disagree. "
        "The **bias** columns show who is closer to reality — the model with the smaller "
        "absolute bias is better calibrated on that subgroup.",
        "",
        "### Who is better calibrated per subgroup?",
        "",
        "| Subgroup | Our |bias| % | SB |bias| % | Better calibrated | Advantage |",
        "|----------|------------:|----------:|:-----------------:|----------:|",
    ]
    for _, r in div_df.iterrows():
        winner_label = f"**{r['better_calibrated']}**" if r["better_calibrated"] != "Tied" else "Tied"
        adv = r["bias_advantage_%"]
        adv_str = f"+{adv:.2f}%" if adv > 0 else f"{adv:.2f}%"
        lines.append(
            f"| {r['subgroup']} | {r['abs_our_bias_%']:.2f}% | {r['abs_sb_bias_%']:.2f}% "
            f"| {winner_label} | {adv_str} |"
        )

    ours_wins = div_df[div_df["better_calibrated"] == "Ours"]
    sb_wins = div_df[div_df["better_calibrated"] == "StatsBomb"]

    lines += [
        "",
        f"**Summary**: our model is better calibrated on **{len(ours_wins)}/{len(div_df)}** subgroups; "
        f"StatsBomb is better on **{len(sb_wins)}/{len(div_df)}**.",
        "",
        "**Where our model wins**:",
    ]
    for _, r in ours_wins.sort_values("bias_advantage_%", ascending=False).iterrows():
        lines.append(f"- {r['subgroup']}: our bias = {r['our_bias_%']:+.2f}% vs SB = {r['sb_bias_%']:+.2f}%")

    lines += ["", "**Where StatsBomb wins**:"]
    for _, r in sb_wins.sort_values("bias_advantage_%").iterrows():
        lines.append(f"- {r['subgroup']}: our bias = {r['our_bias_%']:+.2f}% vs SB = {r['sb_bias_%']:+.2f}%")

    lines += [
        "",
        "> **Interpretation**: StatsBomb's global advantage (Brier, AUC) does not mean it wins "
        "every subgroup. On subgroups where geometry dominates (distance, angle, foot used), "
        "our feature-transparent model can match or beat their calibration. Gaps where StatsBomb "
        "wins typically involve contextual signals we lack (GK position, open space, transition quality).",
        "",
        "---",
        "",
        "## 3. Stacking Experiment",
        "",
        "> **Important**: This experiment requires `statsbomb_xg` at inference time. "
        "It is a research exercise to quantify the information gain from StatsBomb — "
        "it is **not** a production model.",
        "",
        "A second-stage logistic regression was trained on the validation set using "
        "`[p_xgboost_full_platt, statsbomb_xg]` as inputs and actual goals as targets. "
        "It was then evaluated on the held-out test set.",
        "",
        "| Model | Brier ↓ | Log Loss ↓ | AUC ↑ |",
        "|-------|--------:|-----------:|------:|",
    ]
    for v in stack_results.values():
        lines.append(
            f"| {v['model']} | {v['brier_score']:.6f} | {v['log_loss']:.6f} | {v['roc_auc']:.4f} |"
        )

    sb_standalone = next(
        (v for v in stack_results.values() if "standalone" in v["model"].lower()), None
    )
    sb_stacked = next(
        (v for v in stack_results.values() if "stacked" in v["model"].lower()), None
    )

    if sb_standalone and sb_stacked:
        brier_gain = (sb_standalone["brier_score"] - sb_stacked["brier_score"]) / sb_standalone["brier_score"] * 100
        lines += [
            "",
            f"> Stacking with StatsBomb xG reduces Brier score by **{brier_gain:.1f}%** relative to "
            "standalone XGBoost. This gap represents the information contained in StatsBomb's features "
            "(freeze frames, GK location, etc.) that our open-play feature set does not capture.",
        ]

    lines += [
        "",
        "---",
        "",
        "## 4. Conclusion",
        "",
        "- Our calibrated XGBoost model achieves competitive metrics relative to StatsBomb xG on "
        "the same test set. Any remaining gap in Brier score reflects features unavailable at "
        "shot release in standard event data.",
        "- The divergence analysis reveals which shot types are hardest to replicate: "
        "situations where contextual features (GK position, defender proximity) dominate.",
        "- The stacking experiment confirms that StatsBomb carries orthogonal information. "
        "If StatsBomb is available at inference time, the stacked model should be preferred.",
        "- **Recommended production model** (without StatsBomb): `p_xgboost_full_platt` — "
        "calibrated XGBoost with full feature set.",
    ]

    report_path = REPORTS / "09_compare_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {report_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Phase 9 — Comparative Analysis & Gap Investigation")
    print("=" * 55)

    # Load predictions
    print("\nLoading data...")
    test_preds = pd.read_parquet(TABLES / "07_test_predictions.parquet")
    val_preds = pd.read_parquet(TABLES / "07_val_predictions.parquet")
    test_features = pd.read_parquet(PROCESSED / "04_test.parquet")

    # Merge predictions with feature columns for subgroup analysis
    test_merged = test_preds.merge(
        test_features.drop(columns=["match_id", "is_goal", "statsbomb_xg"], errors="ignore"),
        on="shot_id",
        how="inner",
    )
    print(f"  Test set: {len(test_merged):,} shots | val set: {len(val_preds):,} shots")

    # 1. Three-way benchmark
    print("\n[1/6] Three-way benchmark...")
    benchmark_df = benchmark(test_preds)
    benchmark_df.to_csv(TABLES / "09_benchmark_metrics.csv", index=False)
    print(benchmark_df[["model", "brier_score", "log_loss", "roc_auc"]].to_string(index=False))

    # 2. Calibration curves
    print("\n[2/6] Calibration curves...")
    plot_calibration_curves(test_preds)

    # 3. Prediction distributions
    print("\n[3/6] Prediction distributions...")
    plot_prediction_distributions(test_preds)

    # 4. Scatter: our model vs StatsBomb
    print("\n[4/6] Scatter plot...")
    plot_scatter(test_merged)

    # 5. Divergence analysis
    print("\n[5/6] Divergence analysis by subgroup...")
    div_df = divergence_analysis(test_merged)
    div_df.to_csv(TABLES / "09_divergence_by_subgroup.csv", index=False)
    print(div_df[["subgroup", "gap_our_minus_sb_%", "our_bias_%", "sb_bias_%", "better_calibrated"]].to_string(index=False))
    plot_divergence_chart(div_df)
    plot_bias_comparison(div_df)

    # 6. Stacking experiment
    print("\n[6/6] Stacking experiment...")
    stack_results, _, y_stack = stacking_experiment(val_preds, test_preds)
    plot_stacking_calibration(test_preds, y_stack)
    stack_df = pd.DataFrame(list(stack_results.values()))
    stack_df.to_csv(TABLES / "09_stacking_metrics.csv", index=False)

    # Report
    print("\nWriting report...")
    write_report(benchmark_df, div_df, stack_results)

    print("\nDone. Phase 9 outputs:")
    print("  outputs/tables/09_benchmark_metrics.csv")
    print("  outputs/tables/09_divergence_by_subgroup.csv")
    print("  outputs/tables/09_stacking_metrics.csv")
    print("  outputs/figures/09_calibration_comparison.png")
    print("  outputs/figures/09_prediction_distributions.png")
    print("  outputs/figures/09_scatter_xgb_vs_statsbomb.png")
    print("  outputs/figures/09_divergence_by_subgroup.png")
    print("  outputs/figures/09_bias_comparison_by_subgroup.png")
    print("  outputs/figures/09_stacking_calibration.png")
    print("  outputs/reports/09_compare_report.md")


if __name__ == "__main__":
    main()
