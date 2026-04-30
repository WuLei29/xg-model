"""
Phase 4 — Train / Validation / Test Split

Creates a leakage-free, realistic evaluation setup by splitting at the match level
so that every shot from a given match lands in the same fold.

Splitting strategy
------------------
Match-level grouped split with stratification on match-level goal rate.

Why not row-level random split?
  Shots from the same match are correlated — a row-level split would let the model
  see shots from a match in training and then be evaluated on other shots from that
  same match, which is unrealistically easy and inflates generalisation estimates.

Why not chronological split?
  The dataset spans 21 competitions across many seasons with non-sequential season
  IDs.  A clean chronological ordering would require mapping every (competition_id,
  season_id) pair to a calendar year — that mapping is not available in the raw data.
  Match-level grouped split is the recommended default when chronological ordering
  cannot be reliably inferred.

Why stratify?
  The goal rate is ~10.3%, making the target class imbalanced.  Stratification
  ensures that each split has a similar goal rate, preventing one split from being
  systematically harder or easier to evaluate on.

  Stratification procedure:
    1. Compute per-match goal rate.
    2. Bin matches into `n_strata` quantile buckets by goal rate.
    3. Within each bucket, randomly allocate matches to train / val / test in the
       requested proportions (70 / 15 / 15 by default).
    4. Shots inherit their match's split label.

Outputs
-------
data/processed/04_train.parquet          training shots
data/processed/04_val.parquet            validation shots (used for tuning)
data/processed/04_test.parquet           test shots (touched once, for final reporting)
outputs/tables/04_split_index.parquet    shot_id → split label (for joining)
outputs/reports/04_split_report.md       summary statistics and diagnostics

Run
---
    python -m src.data.split
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

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
# Match-level stratified split
# ---------------------------------------------------------------------------

def _compute_match_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per match with shot count and goal rate."""
    return (
        df.groupby("match_id")["is_goal"]
        .agg(n_shots="count", n_goals="sum")
        .assign(goal_rate=lambda x: x["n_goals"] / x["n_shots"])
        .reset_index()
    )


def split_matches(
    match_stats: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    n_strata: int = 5,
    random_state: int = 42,
) -> pd.Series:
    """Assign each match_id to 'train', 'val', or 'test'.

    Uses stratified sampling within goal-rate quantile buckets so that
    each split preserves the overall goal-rate distribution.

    Parameters
    ----------
    match_stats : DataFrame with columns [match_id, goal_rate]
    train_frac  : fraction of matches for training (default 0.70)
    val_frac    : fraction of matches for validation (default 0.15)
    n_strata    : number of quantile buckets (default 5)
    random_state: reproducibility seed

    Returns
    -------
    pd.Series indexed by match_id with values in {'train', 'val', 'test'}
    """
    rng = np.random.default_rng(random_state)

    ms = match_stats.copy()
    ms["stratum"] = pd.qcut(
        ms["goal_rate"],
        q=n_strata,
        labels=False,
        duplicates="drop",
    )

    assignment: dict[int, str] = {}
    for _, group in ms.groupby("stratum"):
        ids = group["match_id"].to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = round(n * train_frac)
        n_val = round(n * val_frac)
        for mid in ids[:n_train]:
            assignment[mid] = "train"
        for mid in ids[n_train : n_train + n_val]:
            assignment[mid] = "val"
        for mid in ids[n_train + n_val :]:
            assignment[mid] = "test"

    return pd.Series(assignment, name="split").rename_axis("match_id")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def verify_no_match_leakage(df: pd.DataFrame) -> None:
    """Raise AssertionError if any match_id appears in more than one split."""
    overlap = (
        df.groupby("match_id")["split"]
        .nunique()
        .pipe(lambda s: s[s > 1])
    )
    assert overlap.empty, (
        f"Leakage detected — {len(overlap)} match(es) appear in multiple splits:\n"
        f"{overlap.index.tolist()}"
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _split_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name in ["train", "val", "test"]:
        subset = df[df["split"] == split_name]
        rows.append(
            {
                "split": split_name,
                "matches": subset["match_id"].nunique(),
                "shots": len(subset),
                "goals": int(subset["is_goal"].sum()),
                "goal_rate_%": round(subset["is_goal"].mean() * 100, 2),
                "pct_of_total": round(len(subset) / len(df) * 100, 1),
            }
        )
    return pd.DataFrame(rows)


def generate_report(df: pd.DataFrame, cfg: dict) -> str:
    sc = cfg["split"]
    stats = _split_stats(df)
    total_matches = df["match_id"].nunique()

    lines: list[str] = ["# Phase 4 — Train / Validation / Test Split Report\n"]

    lines.append("## 1. Strategy\n")
    lines.append(textwrap.dedent(f"""\
        **Method**: match-level grouped split with stratification on per-match goal rate

        Every shot from a given match belongs to exactly one split — this prevents
        the model from memorising match-specific patterns during training and then
        being evaluated on other shots from the same match.

        Stratification bins matches into **{sc['n_strata']} quantile buckets** by goal
        rate, then allocates within each bucket at the requested proportions.  This
        keeps the class imbalance (~10% goals) consistent across splits.

        | Parameter | Value |
        |-----------|-------|
        | Method | match_grouped |
        | Train fraction | {sc['train_frac']:.0%} |
        | Validation fraction | {sc['val_frac']:.0%} |
        | Test fraction | {1 - sc['train_frac'] - sc['val_frac']:.0%} |
        | Stratification strata | {sc['n_strata']} |
        | Random seed | {sc['random_state']} |
    """))

    lines.append("## 2. Split Statistics\n")
    lines.append(stats.to_markdown(index=False))
    lines.append("")
    lines.append(f"Total matches: **{total_matches:,}** | Total shots: **{len(df):,}**\n")

    lines.append("## 3. Leakage Check\n")
    lines.append(
        "**PASS** — no match_id appears in more than one split "
        "(verified programmatically in `verify_no_match_leakage`).\n"
    )

    lines.append("## 4. Why These Splits?\n")
    lines.append(textwrap.dedent("""\
        - **Validation set** (val): used during model selection and hyperparameter tuning.
          Any decision made using val-set metrics is a form of implicit fitting — the val
          set is "used up" as soon as you make a choice based on it.
        - **Test set** (test): touched **once**, only for final reported metrics.
          Comparing multiple models on the test set inflates results (multiple comparisons).
        - **Why not K-fold?** With 3,463 matches and ~82k shots, a single hold-out provides
          enough statistical power.  K-fold would be preferable for much smaller datasets.

        Common pitfall: "I'll just peek at the test set to check for bugs."
        This is fine once during setup, but after any model decision is influenced by test
        metrics, the split must be regenerated (or a new held-out set created).
    """))

    lines.append("## 5. Goal Rate Distribution (Sanity Check)\n")
    lines.append(textwrap.dedent("""\
        The goal rates across splits should be close to the overall 10.3%.
        Large deviations indicate stratification failure or data imbalance.

    """))
    lines.append(stats[["split", "goal_rate_%"]].to_markdown(index=False))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 4 — Train / Validation / Test Split")
    print("=" * 60)

    cfg = _load_cfg("split")
    sc = cfg["split"]

    # Load design matrix
    processed_dir = _dir("data.processed_dir")
    input_path = processed_dir / "03_design_matrix.parquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found: {input_path}. Run `python -m src.features.build` first."
        )

    df = pd.read_parquet(input_path)
    print(f"\nLoaded design matrix : {len(df):,} shots, {df['match_id'].nunique():,} matches")
    print(f"Overall goal rate    : {df['is_goal'].mean()*100:.2f}%")

    # Compute per-match stats and assign split labels
    match_stats = _compute_match_stats(df)
    split_map = split_matches(
        match_stats,
        train_frac=sc["train_frac"],
        val_frac=sc["val_frac"],
        n_strata=sc["n_strata"],
        random_state=sc["random_state"],
    )

    df = df.join(split_map, on="match_id")

    # Verify no match leaks across splits
    verify_no_match_leakage(df)
    print("Leakage check        : PASS — no match appears in multiple splits")

    # Summary statistics
    stats = _split_stats(df)
    print("\nSplit summary:")
    print(stats.to_string(index=False))

    # Save split parquets
    for split_name in ["train", "val", "test"]:
        subset = df[df["split"] == split_name].drop(columns=["split"])
        out_path = processed_dir / f"04_{split_name}.parquet"
        subset.to_parquet(out_path, index=False)
        print(f"\nSaved {split_name:<6}: {out_path.relative_to(ROOT)}  "
              f"({len(subset):,} shots, goal rate {subset['is_goal'].mean()*100:.2f}%)")

    # Save split index (shot_id → split label)
    tables_dir = _dir("outputs.tables_dir")
    index_path = tables_dir / "04_split_index.parquet"
    df[["shot_id", "match_id", "split"]].to_parquet(index_path, index=False)
    print(f"\nSaved split index    : {index_path.relative_to(ROOT)}")

    # Save report
    report = generate_report(df, cfg)
    report_path = _dir("outputs.reports_dir") / "04_split_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved report         : {report_path.relative_to(ROOT)}")

    print("\nPhase 4 complete.")
    print("  Next: python -m src.models.train_logistic")


if __name__ == "__main__":
    main()
