"""
Phase 3 — Feature Engineering

Builds the full feature matrix from the clean Phase 2 samples.
Applies spatial, categorical, contextual, and game-state feature groups.

Feature groups
--------------
baseline_spatial     : distance to goal/posts, visible angle, central flag
baseline_categorical : body-part flags, technique flags, play-pattern flags
baseline_contextual  : is_first_time, is_weak_foot, minute
advanced_gamestate   : score_diff_at_shot, is_late_game

Outputs
-------
data/processed/03_design_matrix.parquet   full feature matrix + targets + metadata
outputs/tables/03_feature_list.json       canonical feature lists per group and set
outputs/reports/03_feature_report.md

Run
---
    python -m src.features.build
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Literal

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


def load_feature_config() -> dict:
    return _load_cfg("features")


def get_feature_list(feature_set: Literal["baseline", "full"] = "baseline") -> list[str]:
    """Return the flat list of feature column names for the given feature set."""
    cfg = load_feature_config()
    groups = cfg["feature_sets"][feature_set]
    features: list[str] = []
    for group in groups:
        features.extend(cfg["features"][group])
    return features


# ---------------------------------------------------------------------------
# Spatial features
# ---------------------------------------------------------------------------

def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives all spatial features from raw (location_x, location_y) coordinates.

    SPADL coordinate system (105 x 68 metres)
    ------------------------------------------
    Goal at x=105, posts at y=30.34 and y=37.66, centered at y=34.
    All distances are in metres.

    Features added
    --------------
    distance_to_goal      distance from shot location to goal center (105, 34)
    visible_angle         angle subtended by the goal posts in degrees —
                          the "opening" the shooter sees; larger = easier chance
    distance_to_near_post distance to the goal post with the closer y-coordinate
    distance_to_far_post  distance to the goal post with the farther y-coordinate
    x_from_goal_line      105 - location_x — how far back from the goal line
    is_central            1 if |y - 34| <= 8.5 (central corridor)
    """
    cfg = load_feature_config()["pitch"]
    GOAL_X: float = cfg["goal_x"]
    GOAL_Y_CENTER: float = cfg["goal_y_center"]
    POST_L: float = cfg["goal_post_left_y"]   # 36.0
    POST_R: float = cfg["goal_post_right_y"]  # 44.0
    CENTRAL_HW: float = cfg["central_half_width"]

    df = df.copy()
    x = df["location_x"].to_numpy(dtype=float)
    y = df["location_y"].to_numpy(dtype=float)

    dx = GOAL_X - x

    # Distance to goal center
    df["distance_to_goal"] = np.sqrt(dx**2 + (GOAL_Y_CENTER - y)**2)

    # Visible goal angle: angle between vectors (shooter → left post) and (shooter → right post).
    # atan2 gives the signed angle of each vector from the x-axis; the difference is the opening angle.
    # For shots inside the pitch (x < 120), dx > 0, so both angles are in (-π/2, π/2) — no wrap risk.
    angle_to_left  = np.arctan2(POST_L - y, dx)
    angle_to_right = np.arctan2(POST_R - y, dx)
    df["visible_angle"] = np.degrees(np.abs(angle_to_right - angle_to_left))

    # Near / far post: the post whose y is closer to the shooter's y
    near_post_y = np.where(y <= GOAL_Y_CENTER, POST_L, POST_R)
    far_post_y  = np.where(y <= GOAL_Y_CENTER, POST_R, POST_L)

    df["distance_to_near_post"] = np.sqrt(dx**2 + (near_post_y - y)**2)
    df["distance_to_far_post"]  = np.sqrt(dx**2 + (far_post_y  - y)**2)

    df["x_from_goal_line"] = dx
    df["is_central"] = (np.abs(y - GOAL_Y_CENTER) <= CENTRAL_HW).astype(int)

    return df


# ---------------------------------------------------------------------------
# Categorical features
# ---------------------------------------------------------------------------

def add_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes shot_body_part, shot_technique, and play_pattern into binary indicator columns.

    Body-part flags (mutually exclusive, exhaustive):
      is_header        body_part == "Head"
      is_right_foot    body_part == "Right Foot"
      is_left_foot     body_part == "Left Foot"
      is_other_bodypart body_part in {"Other", "Unknown"}

    Technique flags (not mutually exclusive with body-part):
      is_normal_technique   technique in {"Normal", "None"}
      is_volley             technique in {"Volley", "Half Volley"}
      is_diving_header      technique == "Diving Header"
      is_exotic_technique   technique in {"Backheel", "Overhead Kick", "Lob"}

    Play-pattern flags (the implicit baseline is all "other" restarts):
      is_regular_play            play_pattern == "Regular Play"
      is_from_counter            play_pattern == "From Counter"
      is_from_set_piece_restart  play_pattern in {"From Corner", "From Free Kick", "From Throw In"}
                                 (shots arising *after* a set piece, not the set piece itself)

    Note on collinearity: for logistic regression, the body-part group is exhaustive
    (all four flags sum to 1), so drop one before fitting (e.g. drop is_other_bodypart
    as the reference category).  Tree models are unaffected.
    """
    df = df.copy()

    # Body part
    bp = df["shot_body_part"]
    df["is_header"]       = (bp == "Head").astype(int)
    df["is_right_foot"]   = (bp == "Right Foot").astype(int)
    df["is_left_foot"]    = (bp == "Left Foot").astype(int)
    df["is_other_bodypart"] = bp.isin({"Other", "Unknown"}).astype(int)

    # Technique
    tech = df["shot_technique"]
    df["is_normal_technique"] = tech.isin({"Normal", "None"}).astype(int)
    df["is_volley"]            = tech.isin({"Volley", "Half Volley"}).astype(int)
    df["is_diving_header"]     = (tech == "Diving Header").astype(int)
    df["is_exotic_technique"]  = tech.isin({"Backheel", "Overhead Kick", "Lob"}).astype(int)

    # Play pattern
    pp = df["play_pattern"]
    df["is_regular_play"]           = (pp == "Regular Play").astype(int)
    df["is_from_counter"]           = (pp == "From Counter").astype(int)
    df["is_from_set_piece_restart"] = pp.isin(
        {"From Corner", "From Free Kick", "From Throw In"}
    ).astype(int)

    return df


# ---------------------------------------------------------------------------
# Contextual features
# ---------------------------------------------------------------------------

def add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures is_first_time and is_weak_foot are integer 0/1 and fills nulls with 0.
    minute is already present; no transformation required.
    """
    df = df.copy()
    for col in ("is_first_time", "is_weak_foot"):
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Game-state features
# ---------------------------------------------------------------------------

def add_gamestate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives game-state features from within the shots table.

    score_diff_at_shot
        For each shot, the goal difference (team_goals - opponent_goals) in that
        match *before* the current shot is taken.  Positive = team is leading.

        Method: sort by (match_id, period, minute, second), then use a shifted
        cumulative sum of is_goal within each (match_id, team_id) group.
        Shifting by 1 ensures the current shot's outcome is excluded.

        Limitation: simultaneous events (same second) have an arbitrary within-second
        ordering; this affects a small fraction of shots and is acceptable.

    is_late_game
        1 if minute > 75 (final 15 minutes + stoppage time).
    """
    df = df.copy()

    df = df.sort_values(
        ["match_id", "period", "minute", "second"],
        kind="mergesort",  # stable sort preserves original row order for ties
    ).reset_index(drop=True)

    # Cumulative goals per team per match, excluding the current shot
    df["_team_goals"] = (
        df.groupby(["match_id", "team_id"])["is_goal"]
        .cumsum()
        .sub(df["is_goal"])
    )

    # Cumulative goals across both teams per match, excluding the current shot
    df["_total_goals"] = (
        df.groupby("match_id")["is_goal"]
        .cumsum()
        .sub(df["is_goal"])
    )

    df["_opponent_goals"] = df["_total_goals"] - df["_team_goals"]
    df["score_diff_at_shot"] = (df["_team_goals"] - df["_opponent_goals"]).astype(int)

    df["is_late_game"] = (df["minute"] > 75).astype(int)

    df = df.drop(columns=["_team_goals", "_total_goals", "_opponent_goals"])
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature groups in order.

    The returned DataFrame contains the original columns plus all new feature
    columns.  Use get_feature_list() to select only the model-input columns.
    """
    df = add_spatial_features(df)
    df = add_categorical_features(df)
    df = add_contextual_features(df)
    df = add_gamestate_features(df)
    return df


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _feature_stats(df: pd.DataFrame, features: list[str], target: str = "is_goal") -> pd.DataFrame:
    rows = []
    for feat in features:
        if feat not in df.columns:
            continue
        s = df[feat]
        row: dict = {
            "feature": feat,
            "dtype": str(s.dtype),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "null_pct": round(float(s.isna().mean() * 100), 2),
        }
        if target in df.columns:
            row["corr_with_target"] = round(float(s.corr(df[target])), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def generate_report(df: pd.DataFrame, cfg: dict) -> str:
    baseline_features = get_feature_list("baseline")
    full_features = get_feature_list("full")
    advanced_only = [f for f in full_features if f not in baseline_features]

    stats = _feature_stats(df, full_features)

    lines = ["# Phase 3 — Feature Engineering Report\n"]

    lines.append("## 1. Feature Groups\n")
    lines.append(textwrap.dedent("""\
        | Group | Features | Description |
        |-------|----------|-------------|
        | baseline_spatial | 6 | Geometric features derived from shot coordinates |
        | baseline_categorical | 11 | Binary encodings of body part, technique, play pattern |
        | baseline_contextual | 3 | Situational flags and match time |
        | advanced_gamestate | 2 | Score context at moment of shot |
        | **Baseline total** | **20** | |
        | **Full total** | **22** | |
    """))

    lines.append("## 2. Leakage Audit\n")
    lines.append(textwrap.dedent("""\
        All features must pass the "available at shot release" test.

        | Feature | Available at shot release? | Reasoning |
        |---------|---------------------------|-----------|
        | distance_to_goal | YES | Computed from shot location, known before kick |
        | visible_angle | YES | Geometry of the pitch — location only |
        | distance_to_near_post | YES | Location only |
        | distance_to_far_post | YES | Location only |
        | x_from_goal_line | YES | Location only |
        | is_central | YES | Location only |
        | is_header / is_right_foot / is_left_foot | YES | Body part chosen at moment of contact |
        | is_normal_technique / is_volley / etc. | YES | Shot technique at moment of contact |
        | is_regular_play / is_from_counter / etc. | YES | How the attack built up before the shot |
        | is_first_time | YES | Whether shooter had time to control the ball |
        | is_weak_foot | YES | Inferred player attribute, not shot-specific outcome |
        | minute | YES | Match clock at shot release |
        | score_diff_at_shot | YES | Cumulative goals *before* this shot — past information |
        | is_late_game | YES | Derived from minute |
    """))

    lines.append("## 3. Spatial Geometry (SPADL Coordinates — metres)\n")
    lines.append(textwrap.dedent(f"""\
        ```
        Pitch: 105 × 68 metres (SPADL standard)
        Goal center:  (105, 34)
        Goal posts:   left=(105, 30.34),  right=(105, 37.66)
        Goal width:   7.32 m

        visible_angle = |atan2(37.66−y, 105−x) − atan2(30.34−y, 105−x)|  [degrees]

        Intuition: a shot from directly in front at 10m sees a ~30° angle;
        a shot from 25m and wide sees only ~5°.
        ```
    """))

    lines.append("## 4. Categorical Encoding Decisions\n")
    lines.append(textwrap.dedent("""\
        **Body part** (4 binary flags — mutually exclusive, sum to 1):
        - For logistic regression: drop `is_other_bodypart` as the reference category
          to avoid perfect multicollinearity.
        - For XGBoost: use all 4 — tree splits handle collinearity natively.

        **Play pattern** (3 flags — implicit baseline is "other restart"):
        - "Other restart" = From Goal Kick, From Keeper, From Kick Off, None, Other
        - Not encoded separately; serves as the reference category for regression.

        **Technique** (4 flags — not exhaustive):
        - No single reference is omitted; "None" technique is grouped into `is_normal_technique`.
        - Overlap with `is_header`: a diving header has `is_header=1` AND `is_diving_header=1`.
          This is intentional — the model can learn from both body-part and technique signals.
    """))

    lines.append("## 5. Game-State Feature: score_diff_at_shot\n")
    lines.append(textwrap.dedent("""\
        Computed by sorting shots within each match (period → minute → second) and
        taking a shifted cumulative sum of `is_goal` per team.  The shift ensures the
        current shot's own outcome is excluded.

        Distribution in the open-play sample:
    """))
    if "score_diff_at_shot" in df.columns:
        counts = df["score_diff_at_shot"].value_counts().sort_index()
        lines.append("| Score diff | Shots | % |")
        lines.append("|-----------|-------|---|")
        for val, cnt in counts.items():
            lines.append(f"| {int(val):+d} | {cnt:,} | {cnt/len(df)*100:.1f}% |")
        lines.append("")
        lines.append(
            "> **Limitation**: simultaneous events sharing the same (period, minute, second) "
            "have an arbitrary intra-second order, affecting a small minority of shots.\n"
        )

    lines.append("## 6. Feature Statistics\n")
    lines.append(stats.to_markdown(index=False))
    lines.append("")

    lines.append("## 7. Feature Sets\n")
    lines.append("**Baseline (20 features)**")
    for f in baseline_features:
        lines.append(f"  - `{f}`")
    lines.append("")
    lines.append("**Advanced additions (2 features)**")
    for f in advanced_only:
        lines.append(f"  - `{f}`")
    lines.append("")

    lines.append("## 8. Design Matrix Summary\n")
    lines.append(f"- Total rows: **{len(df):,}**")
    lines.append(f"- Total columns (features + metadata + targets): **{len(df.columns)}**")
    lines.append(f"- Feature columns (full set): **{len(full_features)}**")
    lines.append(f"- Null values in any feature: **{df[full_features].isna().sum().sum()}**")
    lines.append(f"- Goal rate: **{df['is_goal'].mean()*100:.2f}%**")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 3 — Feature Engineering")
    print("=" * 60)

    cfg = load_feature_config()
    baseline_features = get_feature_list("baseline")
    full_features = get_feature_list("full")

    # Load Phase 2 output
    interim_dir = _dir("data.interim_dir")
    input_path = interim_dir / "02_open_play_shots.parquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input not found: {input_path}. Run `python -m src.data.sample` first."
        )

    df = pd.read_parquet(input_path)
    print(f"\nLoaded: {len(df):,} shots, {len(df.columns)} columns")

    # Build features
    df = build_features(df)
    print(f"Features built:  {len(df.columns)} total columns")

    # Verify no nulls in feature columns
    null_counts = df[full_features].isna().sum()
    null_features = null_counts[null_counts > 0]
    if not null_features.empty:
        print("\n[WARNING] Null values in feature columns:")
        for feat, n in null_features.items():
            print(f"  {feat}: {n} nulls")
    else:
        print("Null check:      PASS — no nulls in any feature column")

    # Print summary statistics
    print(f"\n{'--'*20}")
    print("Spatial feature ranges:")
    for feat in cfg["features"]["baseline_spatial"]:
        s = df[feat]
        print(f"  {feat:<26}: [{s.min():.2f}, {s.max():.2f}]  mean={s.mean():.2f}")

    print("\nBaseline categorical (% = 1):")
    for feat in cfg["features"]["baseline_categorical"]:
        pct = df[feat].mean() * 100
        print(f"  {feat:<32}: {pct:.1f}%")

    print("\nGame-state:")
    print(f"  score_diff_at_shot  range: [{df['score_diff_at_shot'].min()}, {df['score_diff_at_shot'].max()}]  mean={df['score_diff_at_shot'].mean():.2f}")
    print(f"  is_late_game        {df['is_late_game'].mean()*100:.1f}% of shots")

    # Save design matrix
    processed_dir = _dir("data.processed_dir")
    dm_path = processed_dir / "03_design_matrix.parquet"
    df.to_parquet(dm_path, index=False)
    print(f"\nSaved design matrix : {dm_path.relative_to(ROOT)}  ({len(df):,} rows, {len(df.columns)} cols)")

    # Save feature list
    feature_list_payload = {
        "groups": {g: cfg["features"][g] for g in cfg["features"]},
        "sets": {
            name: get_feature_list(name)  # type: ignore[arg-type]
            for name in cfg["feature_sets"]
        },
    }
    tables_dir = _dir("outputs.tables_dir")
    fl_path = tables_dir / "03_feature_list.json"
    fl_path.write_text(
        json.dumps(feature_list_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved feature list  : {fl_path.relative_to(ROOT)}")

    # Save report
    report = generate_report(df, cfg)
    report_path = _dir("outputs.reports_dir") / "03_feature_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved report        : {report_path.relative_to(ROOT)}")

    print("\nPhase 3 complete.")
    print(f"  Baseline feature set : {len(baseline_features)} features")
    print(f"  Full feature set     : {len(full_features)} features")
    print(f"  Next: python -m src.data.split")


if __name__ == "__main__":
    main()
