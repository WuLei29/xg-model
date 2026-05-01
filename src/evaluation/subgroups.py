"""
Phase 8 — Subgroup Analysis

Theory
------
Global metrics (Brier score, AUC) are averages over all shots. They can hide
systematic failures on specific shot types — a model can look good overall while
being badly wrong for headers, long shots, or counter-attacks.

Simpson's paradox in ML
-----------------------
A subgroup with a different base rate and different feature distribution can
have very different model behaviour. Example: headers have a ~10% goal rate
overall but the spatial distribution of headers differs from footed shots
(more crosses from wider positions). A model trained on all shots may apply
the wrong spatial weights when evaluating a header, because it never learned
to condition on header=1 jointly with location.

Conditional calibration
-----------------------
We ask: is the model calibrated *within* each subgroup? That is, when the model
predicts 12% for a header, do ~12% of those headers go in?

A model can be globally calibrated (mean_pred ≈ goal_rate across all shots)
but locally miscalibrated (e.g. it over-predicts headers because it assigns
too much weight to the visible angle, which is typically wide for crossed balls).

Per-subgroup bias = mean_predicted_xg − observed_goal_rate
  Positive → model over-predicts (too optimistic about that shot type)
  Negative → model under-predicts (too pessimistic)

Brier score per subgroup
-----------------------
Each subgroup has its own null Brier score: p̄_sub × (1 − p̄_sub).
Close-range shots have a high base rate (~40%) → null Brier ~0.24.
Long shots have a low base rate (~4%) → null Brier ~0.038.
Comparing raw Brier scores across subgroups is misleading; comparing
"% of null Brier explained" normalises for difficulty.

Common xG model failure modes
-------------------------------
- Headers: body orientation, cross quality, and goalkeeper positioning are not
  captured. The model only knows it's a header + location.
- Long shots: very low conversion rate. Any systematic over-prediction is
  invisible in global metrics (few goals) but large in absolute bias.
- Counter-attacks: the shooting team has advantageous positioning (fewer
  defenders, goalkeeper out of position). The model approximates this with the
  is_from_counter flag but cannot see the actual defensive shape.
- Narrow-angle shots: the visible angle feature captures some of this but
  extremely tight angles (shots from the byline) are geometrically
  near-impossible and the model may not compress probabilities enough.
- First-time shots from crosses: the difficulty depends on the cross height
  and trajectory, which are not in the feature set.

Outputs
-------
outputs/tables/08_subgroup_metrics.csv       full per-subgroup metric table
outputs/figures/08_bias_chart.png            scatter: predicted vs actual rate
outputs/figures/08_brier_by_subgroup.png     Brier reduction vs null per subgroup
outputs/figures/08_xgb_vs_lr.png             XGB vs LR Brier delta per subgroup
outputs/figures/08_calibration_key.png       reliability curves (6 key subgroups)
outputs/reports/08_subgroup_report.md

Run
---
    python -m src.evaluation.subgroups
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]

MIN_SHOTS    = 100   # minimum shots in a subgroup to include it
MIN_GOALS    = 20    # minimum goals in a subgroup for reliable AUC
N_BINS_SMALL = 6     # fewer bins for small subgroups
N_BINS_LARGE = 10


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
# Subgroup definitions
# ---------------------------------------------------------------------------
# Each entry: display_name → (filter_lambda, category_for_colouring)
# filter_lambda takes a DataFrame and returns a boolean Series.

SUBGROUP_DEFS: dict[str, tuple] = {
    # ---- Reference ----
    "All shots":               (lambda df: pd.Series(True, index=df.index),         "overall"),

    # ---- Body part ----
    "Headers":                 (lambda df: df["is_header"] == 1,                    "body_part"),
    "Right foot":              (lambda df: df["is_right_foot"] == 1,                "body_part"),
    "Left foot":               (lambda df: df["is_left_foot"] == 1,                 "body_part"),

    # ---- Technique ----
    "Normal technique":        (lambda df: df["is_normal_technique"] == 1,          "technique"),
    "Volley":                  (lambda df: df["is_volley"] == 1,                    "technique"),
    "Diving header":           (lambda df: df["is_diving_header"] == 1,             "technique"),
    "Exotic technique":        (lambda df: df["is_exotic_technique"] == 1,          "technique"),

    # ---- Context ----
    "First-time":              (lambda df: df["is_first_time"] == 1,                "context"),
    "Not first-time":          (lambda df: df["is_first_time"] == 0,                "context"),
    "Weak foot":               (lambda df: df["is_weak_foot"] == 1,                 "context"),

    # ---- Play pattern ----
    "Regular play":            (lambda df: df["is_regular_play"] == 1,              "pattern"),
    "Counter-attack":          (lambda df: df["is_from_counter"] == 1,              "pattern"),
    "Set-piece restart":       (lambda df: df["is_from_set_piece_restart"] == 1,    "pattern"),

    # ---- Distance ----
    "Close range (<=10m)":     (lambda df: df["distance_to_goal"] <= 10,            "distance"),
    "Medium range (10-20m)":   (lambda df: (df["distance_to_goal"] > 10) &
                                            (df["distance_to_goal"] <= 20),         "distance"),
    "Long range (>20m)":       (lambda df: df["distance_to_goal"] > 20,             "distance"),

    # ---- Visible angle ----
    "Wide angle (>=30 deg)":   (lambda df: df["visible_angle"] >= 30,               "angle"),
    "Medium angle (15-30 deg)":(lambda df: (df["visible_angle"] >= 15) &
                                            (df["visible_angle"] < 30),            "angle"),
    "Narrow angle (<15 deg)":  (lambda df: df["visible_angle"] < 15,                "angle"),

    # ---- Game state ----
    "Winning at shot":         (lambda df: df["score_diff_at_shot"] > 0,            "game_state"),
    "Drawing at shot":         (lambda df: df["score_diff_at_shot"] == 0,           "game_state"),
    "Losing at shot":          (lambda df: df["score_diff_at_shot"] < 0,            "game_state"),
    "Late game (75+ min)":     (lambda df: df["is_late_game"] == 1,                 "game_state"),
}

CATEGORY_COLORS = {
    "overall":    "#333333",
    "body_part":  "#1f77b4",
    "technique":  "#9467bd",
    "context":    "#2ca02c",
    "pattern":    "#d62728",
    "distance":   "#ff7f0e",
    "angle":      "#8c564b",
    "game_state": "#7f7f7f",
}


# ---------------------------------------------------------------------------
# Metrics per subgroup
# ---------------------------------------------------------------------------

def subgroup_metrics(
    df: pd.DataFrame,
    y_col: str,
    p_col: str,
    name: str,
    category: str,
) -> dict | None:
    y = df[y_col].values
    p = df[p_col].values

    n_shots = len(y)
    n_goals = int(y.sum())

    if n_shots < MIN_SHOTS or n_goals < MIN_GOALS:
        return None

    goal_rate    = y.mean()
    mean_pred    = p.mean()
    null_brier   = goal_rate * (1 - goal_rate)
    brier        = brier_score_loss(y, p)
    brier_red    = null_brier - brier
    brier_red_pct = brier_red / null_brier * 100 if null_brier > 0 else 0.0

    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")

    try:
        ll = log_loss(y, p)
    except ValueError:
        ll = float("nan")

    return {
        "subgroup":         name,
        "category":         category,
        "n_shots":          n_shots,
        "n_goals":          n_goals,
        "goal_rate_%":      round(goal_rate * 100, 2),
        "mean_pred_xg_%":   round(mean_pred * 100, 2),
        "bias_%":           round((mean_pred - goal_rate) * 100, 2),
        "null_brier":       round(null_brier, 6),
        "brier_score":      round(brier, 6),
        "brier_reduction":  round(brier_red, 6),
        "brier_red_pct":    round(brier_red_pct, 2),
        "log_loss":         round(ll, 6),
        "roc_auc":          round(auc, 4),
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


def plot_bias_chart(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Scatter: mean predicted xG (x) vs observed goal rate (y) per subgroup.

    Points on the diagonal are perfectly calibrated.
    Points above = model under-predicts that subgroup.
    Points below = model over-predicts that subgroup.
    Bubble size is proportional to sqrt(n_shots).
    """
    df = metrics_df.copy()

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(9, 8))

        for _, row in df.iterrows():
            color  = CATEGORY_COLORS.get(row["category"], "#888888")
            size   = max(40, np.sqrt(row["n_shots"]) * 3)
            marker = "D" if row["subgroup"] == "All shots" else "o"
            ax.scatter(row["mean_pred_xg_%"], row["goal_rate_%"],
                       s=size, color=color, alpha=0.85,
                       marker=marker, zorder=3, edgecolors="white", linewidth=0.5)
            # Offset label to avoid overlap
            ax.annotate(
                row["subgroup"],
                (row["mean_pred_xg_%"], row["goal_rate_%"]),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=7.5,
                color="#333333",
            )

        lim_min = 0
        lim_max = max(df["mean_pred_xg_%"].max(), df["goal_rate_%"].max()) * 1.12
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--",
                linewidth=1, alpha=0.5, label="Perfect calibration", zorder=1)

        ax.set_xlabel("Mean predicted xG (%)")
        ax.set_ylabel("Observed goal rate (%)")
        ax.set_title("Phase 8 — Subgroup Bias: Predicted vs Actual Goal Rate (XGB full, test set)")
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)

        # Legend for categories
        patches = [mpatches.Patch(color=c, label=cat.replace("_", " ").title())
                   for cat, c in CATEGORY_COLORS.items()
                   if cat in df["category"].values]
        ax.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.8)
        ax.grid(True, alpha=0.25)

        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def plot_brier_by_subgroup(metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Horizontal bars: Brier reduction (%) vs null per subgroup, sorted ascending."""
    df = metrics_df[metrics_df["subgroup"] != "All shots"].copy()
    df = df.sort_values("brier_red_pct")

    colors = [CATEGORY_COLORS.get(c, "#888888") for c in df["category"]]

    overall_pct = metrics_df.loc[
        metrics_df["subgroup"] == "All shots", "brier_red_pct"
    ].values[0]

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(9, max(6, len(df) * 0.38)))

        bars = ax.barh(df["subgroup"], df["brier_red_pct"],
                       color=colors, edgecolor="white", linewidth=0.4)

        ax.axvline(overall_pct, color="black", linewidth=1.2, linestyle="--",
                   label=f"Overall ({overall_pct:.1f}%)", zorder=5)

        # Annotate bar values
        for bar, val in zip(bars, df["brier_red_pct"]):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=7.5)

        ax.set_xlabel("Brier score reduction vs null model (%)")
        ax.set_title("Phase 8 — Model Performance by Subgroup\n"
                     "(% of null Brier explained; XGB full, test set)")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_xlim(0, df["brier_red_pct"].max() * 1.15)

        # Category legend
        patches = [mpatches.Patch(color=c, label=cat.replace("_", " ").title())
                   for cat, c in CATEGORY_COLORS.items()
                   if cat in df["category"].values]
        ax.legend(handles=patches + [plt.Line2D([0], [0], color="black",
                  linestyle="--", label=f"Overall ({overall_pct:.1f}%)")],
                  loc="lower right", fontsize=8, framealpha=0.8)

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def plot_xgb_vs_lr(
    metrics_xgb: pd.DataFrame,
    metrics_lr:  pd.DataFrame,
    out_path: Path,
) -> None:
    """Bar chart: XGB Brier delta vs LR Brier delta per subgroup.

    Positive delta = XGB reduces Brier more than LR (XGB wins).
    Negative delta = LR reduces Brier more than XGB (LR wins).
    """
    df = metrics_xgb.merge(
        metrics_lr[["subgroup", "brier_red_pct"]],
        on="subgroup",
        suffixes=("_xgb", "_lr"),
    )
    df["delta"] = df["brier_red_pct_xgb"] - df["brier_red_pct_lr"]
    df = df[df["subgroup"] != "All shots"].sort_values("delta")

    colors = ["#ff7f0e" if d >= 0 else "#1f77b4" for d in df["delta"]]

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(9, max(6, len(df) * 0.38)))

        ax.barh(df["subgroup"], df["delta"], color=colors,
                edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=0.8)

        ax.set_xlabel("XGB Brier reduction % − LR Brier reduction %\n"
                      "(positive = XGB better, negative = LR better)")
        ax.set_title("Phase 8 — XGBoost vs Logistic Regression by Subgroup\n"
                     "(full feature set, test set)")

        patches = [
            mpatches.Patch(color="#ff7f0e", label="XGBoost better"),
            mpatches.Patch(color="#1f77b4", label="Logistic better"),
        ]
        ax.legend(handles=patches, fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def plot_calibration_key(
    test_df: pd.DataFrame,
    y_col: str,
    p_col: str,
    key_subgroups: list[str],
    out_path: Path,
) -> None:
    """2×3 reliability diagrams for six key subgroups."""
    n_cols = 3
    n_rows = (len(key_subgroups) + n_cols - 1) // n_cols
    _FMT = mticker.PercentFormatter(xmax=1, decimals=0)

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(12, n_rows * 3.8),
                                 sharex=False, sharey=False)
        axes_flat = axes.flatten()

        for ax, sg_name in zip(axes_flat, key_subgroups):
            fn, category = SUBGROUP_DEFS[sg_name]
            mask = fn(test_df)
            sub  = test_df[mask]

            y = sub[y_col].values
            p = sub[p_col].values

            if len(y) < MIN_SHOTS or y.sum() < MIN_GOALS:
                ax.set_title(f"{sg_name}\n(insufficient data)")
                ax.axis("off")
                continue

            n_bins = N_BINS_SMALL if len(y) < 500 else N_BINS_LARGE
            try:
                frac, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
                ax.plot(mean_pred, frac, "s-", color=CATEGORY_COLORS.get(category, "#888"),
                        linewidth=2, markersize=5)
            except ValueError:
                ax.set_title(f"{sg_name}\n(calibration error)")
                ax.axis("off")
                continue

            goal_rate = y.mean()
            brier     = brier_score_loss(y, p)
            mean_pred_val = p.mean()

            ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
            ax.set_title(
                f"{sg_name}\n"
                f"n={len(y):,}  goal_rate={goal_rate*100:.1f}%  "
                f"pred={mean_pred_val*100:.1f}%  Brier={brier:.4f}",
                fontsize=9,
            )
            ax.xaxis.set_major_formatter(_FMT)
            ax.yaxis.set_major_formatter(_FMT)
            ax.grid(True, alpha=0.3)

            # Upper xlim based on max observed probability (avoid huge empty space)
            xlim = min(max(p) * 1.1, 1.0)
            ax.set_xlim(0, xlim)
            ax.set_ylim(0, xlim)

        # Hide any unused axes
        for ax in axes_flat[len(key_subgroups):]:
            ax.axis("off")

        fig.supxlabel("Mean predicted probability", y=0.01)
        fig.supylabel("Observed goal rate", x=0.01)
        fig.suptitle("Phase 8 — Calibration by Key Subgroup (XGB full, test set)", fontsize=13)
        fig.tight_layout(rect=[0.03, 0.03, 1, 0.97])
        fig.savefig(out_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(metrics_xgb: pd.DataFrame, metrics_lr: pd.DataFrame) -> str:
    lines: list[str] = ["# Phase 8 — Subgroup Analysis\n"]

    lines.append("## 1. Theory\n")
    lines.append(textwrap.dedent("""\
        ### Why subgroup analysis matters

        A model with good global metrics can still be systematically wrong for
        specific shot types. This is a version of Simpson's paradox: aggregating
        across subgroups with different base rates can mask local failures.

        For xG specifically, the risk is *conditional miscalibration*: the model
        is globally well-calibrated (overall mean pred ≈ overall goal rate) but
        over-predicts one shot type and under-predicts another in a way that cancels
        in aggregate. This would cause player and team xG comparisons to be
        misleading for certain playing styles (e.g. a team that crosses a lot and
        produces many headed shots).

        ### Per-subgroup null Brier score

        Each subgroup has its own difficulty level. Close-range shots have a ~40%
        goal rate → null Brier ≈ 0.24. Long shots have ~4% → null Brier ≈ 0.04.
        The metric **"% of null Brier explained"** normalises for this:

            brier_reduction% = (null_brier − model_brier) / null_brier × 100

        A higher % means the model is outperforming the "just predict the subgroup
        base rate" baseline by a larger margin.

        ### Bias direction

        bias = mean_predicted_xg − observed_goal_rate

        - Positive bias: model over-predicts (says 12%, reality is 9%).
        - Negative bias: model under-predicts (says 8%, reality is 11%).
        - Near-zero bias: model is locally well-calibrated.

    """))

    lines.append("## 2. Results Table (XGBoost full, test set)\n")
    lines.append(metrics_xgb.to_markdown(index=False))
    lines.append("")

    lines.append("## 3. Failure Mode Analysis\n")

    # Identify worst subgroups by absolute bias
    valid = metrics_xgb[metrics_xgb["subgroup"] != "All shots"].copy()
    valid["abs_bias"] = valid["bias_%"].abs()
    top_bias = valid.nlargest(5, "abs_bias")

    lines.append("### Largest calibration errors (by |bias|)\n")
    for _, row in top_bias.iterrows():
        direction = "over-predicts" if row["bias_%"] > 0 else "under-predicts"
        lines.append(
            f"- **{row['subgroup']}** ({row['n_shots']:,} shots, "
            f"{row['goal_rate_%']:.1f}% actual): model {direction} by "
            f"**{abs(row['bias_%']):.2f}pp** "
            f"(pred={row['mean_pred_xg_%']:.1f}%, actual={row['goal_rate_%']:.1f}%)\n"
        )

    # Subgroups where XGB beats LR most
    merged = metrics_xgb.merge(
        metrics_lr[["subgroup", "brier_red_pct"]],
        on="subgroup", suffixes=("_xgb", "_lr")
    )
    merged["delta"] = merged["brier_red_pct_xgb"] - merged["brier_red_pct_lr"]
    top_xgb = merged[merged["subgroup"] != "All shots"].nlargest(3, "delta")
    top_lr  = merged[merged["subgroup"] != "All shots"].nsmallest(3, "delta")

    lines.append("### Subgroups where XGBoost gains most over Logistic Regression\n")
    for _, row in top_xgb.iterrows():
        lines.append(
            f"- **{row['subgroup']}**: XGB Brier reduction {row['brier_red_pct_xgb']:.1f}% "
            f"vs LR {row['brier_red_pct_lr']:.1f}% "
            f"(delta = +{row['delta']:.1f}pp)\n"
        )

    lines.append("### Subgroups where Logistic Regression matches or beats XGBoost\n")
    for _, row in top_lr.iterrows():
        lines.append(
            f"- **{row['subgroup']}**: LR Brier reduction {row['brier_red_pct_lr']:.1f}% "
            f"vs XGB {row['brier_red_pct_xgb']:.1f}% "
            f"(delta = {row['delta']:.1f}pp)\n"
        )

    lines.append("## 4. Key Findings\n")
    lines.append(textwrap.dedent("""\
        See figures in `outputs/figures/08_*.png` for visualisations.

        **What to look for in the bias chart (08_bias_chart.png):**
        - Points far above the diagonal: model is under-predicting that shot type
          (potentially missing context that makes those shots more dangerous).
        - Points far below the diagonal: model is over-predicting (assigning too high
          an xG to shots that are harder than the features suggest).
        - The size of each bubble reflects the sample size — large bubbles off the
          diagonal represent the most impactful systematic errors.

        **Interpreting the Brier reduction chart (08_brier_by_subgroup.png):**
        - Subgroups below the overall line are where the model struggles most
          relative to its difficulty level.
        - Very low Brier reduction on long-range shots is expected — there is very
          little predictive signal at extreme distances.

        **XGBoost vs Logistic Regression (08_xgb_vs_lr.png):**
        - XGBoost gains most where interactions matter (distance × angle × body part).
        - Logistic regression is competitive or better for subgroups with linear
          relationships (e.g. pure distance-driven subgroups).
    """))

    lines.append("## 5. Implications for Production\n")
    lines.append(textwrap.dedent("""\
        When using this model operationally:
        - Interpret xG totals for header-heavy teams (set-piece specialists,
          long-ball teams) with extra caution — the model's header estimates
          are driven by location only, missing cross quality and aerial ability.
        - Long-range shot counts in xG totals are nearly noise — the model cannot
          distinguish a speculative long shot from a well-struck one beyond the
          basic spatial features.
        - Counter-attack xG may be slightly under-estimated — the defensive
          disorganisation context is approximated by a single flag, not the
          actual positions of defenders.

        These limitations are fundamental to event-only data; they would require
        freeze-frame features (goalkeeper position, defender distances) to resolve.
    """))

    lines.append("## 6. Next Steps\n")
    lines.append(textwrap.dedent("""\
        Phase 9 — Teacher Imitation Model: train a separate model (Track B) to
        approximate StatsBomb's xG values directly. Comparing the imitation model's
        subgroup errors to our model's will reveal what contextual information
        StatsBomb's model captures that ours cannot.
    """))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 8 — Subgroup Analysis")
    print("=" * 60)

    tables_dir  = _dir("outputs.tables_dir")
    figures_dir = _dir("outputs.figures_dir")
    reports_dir = _dir("outputs.reports_dir")

    # Load test split (features + metadata) and predictions
    processed_dir = ROOT / _load_cfg("paths")["data"]["processed_dir"]
    test_feat  = pd.read_parquet(processed_dir / "04_test.parquet")
    test_preds = pd.read_parquet(tables_dir / "07_test_predictions.parquet")

    # Merge predictions onto feature frame (align by shot_id)
    test_df = test_feat.merge(
        test_preds[["shot_id", "p_xgboost_full", "p_logistic_full"]],
        on="shot_id", how="left",
    )

    print(f"\nTest set: {len(test_df):,} shots | goal rate {test_df['is_goal'].mean()*100:.2f}%")
    print(f"Subgroups defined: {len(SUBGROUP_DEFS)}")

    # ---------------------------------------------------------------------------
    # Compute metrics for XGBoost full and Logistic full
    # ---------------------------------------------------------------------------
    metrics_xgb: list[dict] = []
    metrics_lr:  list[dict] = []
    skipped = 0

    for sg_name, (fn, category) in SUBGROUP_DEFS.items():
        mask = fn(test_df)
        sub  = test_df[mask]

        row_xgb = subgroup_metrics(sub, "is_goal", "p_xgboost_full",  sg_name, category)
        row_lr  = subgroup_metrics(sub, "is_goal", "p_logistic_full", sg_name, category)

        if row_xgb is None:
            skipped += 1
            print(f"  Skipped (too small): {sg_name} ({mask.sum()} shots)")
            continue

        metrics_xgb.append(row_xgb)
        if row_lr is not None:
            metrics_lr.append(row_lr)

    print(f"\nSubgroups evaluated : {len(metrics_xgb)}")
    print(f"Subgroups skipped   : {skipped} (< {MIN_SHOTS} shots or < {MIN_GOALS} goals)")

    df_xgb = pd.DataFrame(metrics_xgb)
    df_lr  = pd.DataFrame(metrics_lr)

    # ---------------------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------------------
    display_cols = ["subgroup", "n_shots", "goal_rate_%", "mean_pred_xg_%",
                    "bias_%", "brier_red_pct", "roc_auc"]
    print("\nXGBoost full — per-subgroup metrics (test set):")
    print(df_xgb[display_cols].to_string(index=False))

    # ---------------------------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------------------------
    df_xgb.to_csv(tables_dir / "08_subgroup_metrics_xgb.csv", index=False)
    df_lr.to_csv(tables_dir  / "08_subgroup_metrics_lr.csv",  index=False)
    print(f"\nSaved metrics to {tables_dir.relative_to(ROOT)}/")

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------
    plot_bias_chart(df_xgb, figures_dir / "08_bias_chart.png")
    print(f"Saved bias chart        : outputs/figures/08_bias_chart.png")

    plot_brier_by_subgroup(df_xgb, figures_dir / "08_brier_by_subgroup.png")
    print(f"Saved Brier by subgroup : outputs/figures/08_brier_by_subgroup.png")

    if len(df_lr):
        plot_xgb_vs_lr(df_xgb, df_lr, figures_dir / "08_xgb_vs_lr.png")
        print(f"Saved XGB vs LR         : outputs/figures/08_xgb_vs_lr.png")

    key_subgroups = [
        "Headers",
        "Close range (<=10m)",
        "Long range (>20m)",
        "Counter-attack",
        "First-time",
        "Narrow angle (<15 deg)",
    ]
    plot_calibration_key(
        test_df, "is_goal", "p_xgboost_full",
        key_subgroups,
        figures_dir / "08_calibration_key.png",
    )
    print(f"Saved calibration grid  : outputs/figures/08_calibration_key.png")

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    report = generate_report(df_xgb, df_lr)
    (reports_dir / "08_subgroup_report.md").write_text(report, encoding="utf-8")
    print(f"Saved report            : outputs/reports/08_subgroup_report.md")

    # ---------------------------------------------------------------------------
    # Highlight biggest biases
    # ---------------------------------------------------------------------------
    print("\n--- Top 5 subgroups by absolute bias ---")
    top = df_xgb[df_xgb["subgroup"] != "All shots"].copy()
    top["abs_bias"] = top["bias_%"].abs()
    top = top.nlargest(5, "abs_bias")[["subgroup", "n_shots", "goal_rate_%",
                                       "mean_pred_xg_%", "bias_%"]]
    print(top.to_string(index=False))

    print("\nPhase 8 complete.")
    print("  Next: python -m src.models.train_teacher  (Phase 9)")


if __name__ == "__main__":
    main()
