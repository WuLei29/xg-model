"""
Phase 2 -- Sample Definition

Filters the raw dataset to a clean, well-defined modeling population:
  - StatsBomb data only (drops Wyscout -- richer features, consistent schema)
  - Separate parquets for open-play shots and direct free-kick shots
  - Infers preferred foot per player; adds `is_weak_foot` feature
  - Drops all post-shot / forbidden columns

Outputs:
  data/interim/02_open_play_shots.parquet   -- main modeling dataset (~80k shots)
  data/interim/02_free_kick_shots.parquet   -- separate free-kick dataset (~4k shots)
  data/interim/02_preferred_foot.parquet    -- player-level preferred foot lookup
  outputs/reports/02_sample_report.md
  outputs/tables/02_sample_stats.csv

Run:
    python -m src.data.sample
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

# Forbidden post-shot columns -- dropped in all outputs
FORBIDDEN_COLS = ["end_location_x", "end_location_y"]

# shot_type values to treat as separate populations (not open play)
EXCLUDED_TYPES = {"Penalty", "Free Kick", "Corner", "Kick Off"}
FREE_KICK_TYPE = "Free Kick"

PREFERRED_FOOT_THRESHOLD = 0.53 # fraction to call a foot "preferred" (fallback rule)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_paths() -> dict:
    with open(ROOT / "configs" / "paths.yaml") as f:
        return yaml.safe_load(f)


def _dir(key: str) -> Path:
    cfg = _load_paths()
    keys = key.split(".")
    node = cfg
    for k in keys:
        node = node[k]
    p = ROOT / node
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    raw_path = ROOT / _load_paths()["data"]["raw_dir"] / "shots_raw.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. Run `python -m src.data.audit` first."
        )
    return pd.read_parquet(raw_path)


# ---------------------------------------------------------------------------
# Preferred foot inference
# ---------------------------------------------------------------------------

def infer_preferred_foot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three-rule priority system for preferred foot inference:

    1. Penalties: if a player has taken any, the majority foot used = preferred.
       Rationale: players virtually always use their preferred foot for penalties.
    2. Free kicks: if no penalties, the majority foot used on direct free kicks = preferred.
       Same rationale -- set-piece specialists shoot with their dominant foot.
    3. Fallback: foot used on >= PREFERRED_FOOT_THRESHOLD of all footed shots -> preferred.
       If no clear majority -> 'Ambiguous'.

    Returns a DataFrame with columns:
      player_id, preferred_foot, source, left_pct, right_pct, n_footed_shots
    where `source` is one of: 'penalty', 'free_kick', 'threshold', 'ambiguous'.
    """
    footed_types = ["Left Foot", "Right Foot"]

    def _majority_foot(series: pd.Series) -> str | None:
        counts = series.value_counts()
        return counts.index[0] if len(counts) > 0 else None

    # Rule 1: penalties
    pen = df[(df["shot_type"] == "Penalty") & df["shot_body_part"].isin(footed_types)]
    pen_pref = (
        pen.groupby("player_id")["shot_body_part"]
        .apply(_majority_foot)
        .rename("pen_foot")
    )

    # Rule 2: free kicks
    fk = df[(df["shot_type"] == "Free Kick") & df["shot_body_part"].isin(footed_types)]
    fk_pref = (
        fk.groupby("player_id")["shot_body_part"]
        .apply(_majority_foot)
        .rename("fk_foot")
    )

    # Rule 3: threshold fallback on all footed shots
    footed = df[df["shot_body_part"].isin(footed_types)]
    counts = (
        footed.groupby(["player_id", "shot_body_part"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"Left Foot": "n_left", "Right Foot": "n_right"})
    )
    counts["n_footed_shots"] = counts["n_left"] + counts["n_right"]
    counts["left_pct"] = counts["n_left"] / counts["n_footed_shots"]
    counts["right_pct"] = counts["n_right"] / counts["n_footed_shots"]

    def _threshold_foot(row: pd.Series) -> str:
        if row["left_pct"] >= PREFERRED_FOOT_THRESHOLD:
            return "Left Foot"
        if row["right_pct"] >= PREFERRED_FOOT_THRESHOLD:
            return "Right Foot"
        return "Ambiguous"

    counts["threshold_foot"] = counts.apply(_threshold_foot, axis=1)

    # Combine with priority: penalty > free_kick > threshold
    result = counts.reset_index()[["player_id", "n_left", "n_right", "n_footed_shots", "left_pct", "right_pct", "threshold_foot"]]
    result = result.merge(pen_pref, on="player_id", how="left")
    result = result.merge(fk_pref, on="player_id", how="left")

    def _resolve(row: pd.Series) -> tuple[str, str]:
        if pd.notna(row.get("pen_foot")):
            return row["pen_foot"], "penalty"
        if pd.notna(row.get("fk_foot")):
            return row["fk_foot"], "free_kick"
        if row["threshold_foot"] != "Ambiguous":
            return row["threshold_foot"], "threshold"
        return "Ambiguous", "ambiguous"

    resolved = result.apply(_resolve, axis=1, result_type="expand")
    result["preferred_foot"] = resolved[0]
    result["source"] = resolved[1]

    return result[[
        "player_id", "preferred_foot", "source",
        "left_pct", "right_pct", "n_footed_shots",
    ]]


def add_weak_foot(df: pd.DataFrame, preferred_foot: pd.DataFrame, allow_weak_foot: bool = True) -> pd.DataFrame:
    """
    Adds `is_weak_foot` column.

    allow_weak_foot=False: always sets is_weak_foot=0 (used for free kicks, where the
    preferred foot rule is derived from those same shots -- flagging weak foot there
    would be circular and meaningless).
    """
    merged = df.merge(preferred_foot[["player_id", "preferred_foot"]], on="player_id", how="left")
    merged["preferred_foot"] = merged["preferred_foot"].fillna("Ambiguous")

    if allow_weak_foot:
        footed_mask = merged["shot_body_part"].isin(["Left Foot", "Right Foot"])
        known_pref_mask = merged["preferred_foot"].isin(["Left Foot", "Right Foot"])
        weak_mask = footed_mask & known_pref_mask & (merged["shot_body_part"] != merged["preferred_foot"])
        merged["is_weak_foot"] = weak_mask.astype(int)
    else:
        merged["is_weak_foot"] = 0

    return merged.drop(columns=["preferred_foot"])


# ---------------------------------------------------------------------------
# Sample building
# ---------------------------------------------------------------------------

SB_LENGTH = 120.0
SB_WIDTH = 80.0
SPADL_LENGTH = 105.0
SPADL_WIDTH = 68.0


def _convert_coordinates_to_spadl(df: pd.DataFrame) -> pd.DataFrame:
    """Convert StatsBomb coordinates (120x80) to SPADL standard (105x68 metres)."""
    df = df.copy()
    df["location_x"] = df["location_x"] * (SPADL_LENGTH / SB_LENGTH)
    df["location_y"] = df["location_y"] * (SPADL_WIDTH / SB_WIDTH)
    return df


def build_samples(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (open_play_df, free_kick_df), both StatsBomb-only and post-shot-columns dropped.
    Coordinates are converted from StatsBomb (120x80) to SPADL (105x68 metres).
    """
    # StatsBomb only: identified by non-null statsbomb_xg
    statsbomb = df[df["statsbomb_xg"].notna()].copy()

    # Drop forbidden post-shot columns
    statsbomb = statsbomb.drop(columns=[c for c in FORBIDDEN_COLS if c in statsbomb.columns])

    # Convert coordinates from StatsBomb to SPADL
    statsbomb = _convert_coordinates_to_spadl(statsbomb)

    # Open play: everything that's not a penalty, free kick, corner, or kick-off
    open_play = statsbomb[~statsbomb["shot_type"].isin(EXCLUDED_TYPES)].copy()

    # Direct free kicks: separate population
    free_kicks = statsbomb[statsbomb["shot_type"] == FREE_KICK_TYPE].copy()

    return open_play, free_kicks


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def _sample_stats(df: pd.DataFrame, label: str) -> dict:
    goals = int(df["is_goal"].sum())
    n = len(df)
    return {
        "sample": label,
        "n_shots": n,
        "n_goals": goals,
        "n_non_goals": n - goals,
        "goal_rate_pct": round(goals / n * 100, 2) if n > 0 else 0.0,
        "n_matches": int(df["match_id"].nunique()),
        "n_players": int(df["player_id"].nunique()),
        "n_weak_foot": int(df["is_weak_foot"].sum()) if "is_weak_foot" in df.columns else None,
        "weak_foot_pct": (
            round(df["is_weak_foot"].mean() * 100, 2)
            if "is_weak_foot" in df.columns else None
        ),
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_report(
    raw_n: int,
    open_play: pd.DataFrame,
    free_kicks: pd.DataFrame,
    preferred_foot: pd.DataFrame,
) -> str:
    op = _sample_stats(open_play, "Open Play")
    fk = _sample_stats(free_kicks, "Free Kick")

    pref_counts = preferred_foot["preferred_foot"].value_counts()

    lines = ["# Phase 2 -- Sample Definition Report\n"]

    lines.append("## 1. Inclusion Criteria\n")
    lines.append(textwrap.dedent("""\
        | Decision | Choice | Rationale |
        |----------|--------|-----------|
        | Data source | StatsBomb only | Richer features (technique, play pattern, is_first_time); consistent schema; required for Track B (provider xG available) |
        | Penalties | Excluded | Near-deterministic outcome; fundamentally different generative process |
        | Direct free kicks | Separate dataset | Different shooting geometry (wall, set-piece setup); too different to pool with open play |
        | Corners | Excluded | 28 samples -- too few; different generative process |
        | Kick-offs | Excluded | 1 sample |
        | Wyscout shots | Excluded | Missing technique/body-part/play-pattern; no StatsBomb xG for Track B |
    """))

    lines.append("## 2. Population Summary\n")
    lines.append(f"Raw dataset: **{raw_n:,}** shots total\n")
    lines.append("| Sample | Shots | Goals | Goal rate | Matches | Players |")
    lines.append("|--------|-------|-------|-----------|---------|---------|")
    for s in [op, fk]:
        lines.append(
            f"| {s['sample']} | {s['n_shots']:,} | {s['n_goals']:,} | "
            f"{s['goal_rate_pct']}% | {s['n_matches']:,} | {s['n_players']:,} |"
        )
    lines.append("")

    lines.append("## 3. Preferred Foot Inference\n")
    lines.append(textwrap.dedent(f"""\
        Three-rule priority system:
        1. **Penalty foot** — if a player has taken any penalties, the majority foot used = preferred
        2. **Free kick foot** — if no penalties, the majority foot used on direct free kicks = preferred
        3. **Threshold fallback** — foot used on ≥{int(PREFERRED_FOOT_THRESHOLD*100)}% of all footed shots = preferred; otherwise Ambiguous
    """))
    lines.append("| Preferred foot | Players |")
    lines.append("|----------------|---------|")
    for cat in ["Right Foot", "Left Foot", "Ambiguous"]:
        cnt = pref_counts.get(cat, 0)
        lines.append(f"| {cat} | {cnt:,} |")
    lines.append("")
    src_counts = preferred_foot["source"].value_counts()
    lines.append("| Inference rule | Players |")
    lines.append("|----------------|---------|")
    for src in ["penalty", "free_kick", "threshold", "ambiguous"]:
        cnt = src_counts.get(src, 0)
        lines.append(f"| {src} | {cnt:,} |")
    lines.append("")

    lines.append("## 4. Weak Foot Feature\n")
    lines.append(
        "`is_weak_foot = 1` when a footed shot is taken with the non-preferred foot "
        "AND the player has a known preference (not Ambiguous).\n"
    )
    lines.append("| Sample | Weak foot shots | % of footed shots |")
    lines.append("|--------|-----------------|-------------------|")
    for s in [op, fk]:
        lines.append(
            f"| {s['sample']} | {s['n_weak_foot']:,} | {s['weak_foot_pct']}% |"
        )
    lines.append("")

    lines.append("## 5. Final Column Set\n")
    lines.append(
        "Forbidden post-shot columns (`end_location_x`, `end_location_y`) are **dropped** "
        "from all outputs. `shot_outcome` is retained for analysis/subgroup work only -- "
        "never a feature.\n"
    )

    op_cols = list(open_play.columns)
    lines.append("**Open play columns:**")
    for c in op_cols:
        lines.append(f"  - `{c}`")
    lines.append("")

    lines.append("## 6. Track Populations\n")
    lines.append(textwrap.dedent(f"""\
        **Track A (Real xG)**
        - File: `data/interim/02_open_play_shots.parquet`
        - Population: {op['n_shots']:,} open-play StatsBomb shots
        - Target: `is_goal`

        **Track B (Teacher Imitation)**
        - File: same as Track A (`statsbomb_xg` column is the target)
        - Population: same {op['n_shots']:,} shots (all have `statsbomb_xg`)
        - Target: `statsbomb_xg`

        **Free Kick model (future)**
        - File: `data/interim/02_free_kick_shots.parquet`
        - Population: {fk['n_shots']:,} direct free-kick shots
        - Note: lightweight model (logistic regression) recommended given sample size
    """))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 2 -- Sample Definition")
    print("=" * 60)

    df = load_raw()
    print(f"\nRaw dataset: {len(df):,} rows")

    # Infer preferred foot from all StatsBomb footed shots (before population filter)
    statsbomb_all = df[df["statsbomb_xg"].notna()].copy()
    preferred_foot = infer_preferred_foot(statsbomb_all)

    open_play, free_kicks = build_samples(df)
    open_play = add_weak_foot(open_play, preferred_foot, allow_weak_foot=True)
    free_kicks = add_weak_foot(free_kicks, preferred_foot, allow_weak_foot=False)

    # Print summary
    print(f"\n{'--'*20}")
    for label, sample in [("Open play", open_play), ("Free kicks", free_kicks)]:
        n = len(sample)
        goals = sample["is_goal"].sum()
        wf = sample["is_weak_foot"].sum()
        print(f"{label:<12}: {n:,} shots | {goals:,} goals ({goals/n*100:.1f}%) | "
              f"{wf:,} weak-foot ({wf/n*100:.1f}%)")

    pref = preferred_foot["preferred_foot"].value_counts()
    print(f"\nPreferred foot (StatsBomb players):")
    for cat in ["Right Foot", "Left Foot", "Ambiguous"]:
        print(f"  {cat:<12}: {pref.get(cat, 0):,}")

    # Save parquets
    interim_dir = _dir("data.interim_dir")
    open_play_path = interim_dir / "02_open_play_shots.parquet"
    free_kick_path = interim_dir / "02_free_kick_shots.parquet"
    pref_foot_path = interim_dir / "02_preferred_foot.parquet"

    open_play.to_parquet(open_play_path, index=False)
    free_kicks.to_parquet(free_kick_path, index=False)
    preferred_foot.to_parquet(pref_foot_path, index=False)

    print(f"\nSaved:")
    print(f"  {open_play_path.relative_to(ROOT)}  ({len(open_play):,} rows)")
    print(f"  {free_kick_path.relative_to(ROOT)}  ({len(free_kicks):,} rows)")
    print(f"  {pref_foot_path.relative_to(ROOT)}  ({len(preferred_foot):,} players)")

    # Save report
    report = generate_report(len(df), open_play, free_kicks, preferred_foot)
    report_path = _dir("outputs.reports_dir") / "02_sample_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  {report_path.relative_to(ROOT)}")

    # Save stats CSV
    stats = pd.DataFrame([
        _sample_stats(open_play, "Open Play"),
        _sample_stats(free_kicks, "Free Kick"),
    ])
    stats_path = _dir("outputs.tables_dir") / "02_sample_stats.csv"
    stats.to_csv(stats_path, index=False)
    print(f"  {stats_path.relative_to(ROOT)}")

    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
