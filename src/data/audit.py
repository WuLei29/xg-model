"""
Phase 1 -- Data Audit & Problem Definition

Downloads the raw shot dataset from HuggingFace (if not already present),
performs a full schema and quality audit, and saves:
  - outputs/reports/01_audit_report.md
  - outputs/tables/01_value_mapping.json
  - outputs/tables/01_null_summary.csv

Run:
    python -m src.data.audit
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")


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

def load_shot_data() -> pd.DataFrame:
    """Load shot data. Uses local parquet cache if available."""
    raw_dir = _dir("data.raw_dir")
    local = raw_dir / "shots_raw.parquet"

    if local.exists():
        print(f"Loading local cache: {local}")
        return pd.read_parquet(local)

    print("Downloading shot data from HuggingFace...")
    cfg = _load_paths()
    from datasets import load_dataset  # type: ignore
    ds = load_dataset(cfg["data"]["huggingface_shot_data"], split="train")
    df = ds.to_pandas()
    df.to_parquet(local, index=False)
    print(f"Saved {len(df):,} rows to {local}")
    return df


# ---------------------------------------------------------------------------
# Column roles for the xg-shot-data schema
# ---------------------------------------------------------------------------

COLUMN_ROLES: dict[str, str] = {
    "shot_id":           "identifier",
    "match_key":         "identifier",
    "match_id":          "match_metadata",
    "competition_id":    "match_metadata",
    "season_id":         "match_metadata",
    "player_id":         "match_metadata",
    "team_id":           "match_metadata",
    "period":            "match_metadata",
    "minute":            "candidate_feature",
    "second":            "match_metadata",
    "location_x":        "candidate_feature",
    "location_y":        "candidate_feature",
    "distance_to_goal":  "candidate_feature",
    "shot_angle":        "candidate_feature",
    "end_location_x":    "forbidden_post_shot",
    "end_location_y":    "forbidden_post_shot",
    "is_goal":           "target_track_a",
    "shot_outcome":      "target_derived",
    "statsbomb_xg":      "target_track_b",
    "shot_body_part":    "candidate_feature",
    "shot_technique":    "candidate_feature",
    "shot_type":         "candidate_feature",
    "play_pattern":      "candidate_feature",
    "is_first_time":     "candidate_feature",
}

FORBIDDEN_REASONS: dict[str, str] = {
    "end_location_x": (
        "Post-shot: reveals where the ball landed after the shot. "
        "Only observable after the shot is taken."
    ),
    "end_location_y": (
        "Post-shot: same as end_location_x. "
        "Ball trajectory is unknown at shot release."
    ),
    "shot_outcome": (
        "This IS the outcome (Goal/Saved/Blocked/Off T). "
        "Using it as a feature would be perfect leakage."
    ),
    "statsbomb_xg": (
        "In Track A this is a provider estimate, not ground truth. "
        "It is the target for Track B. Never a Track A feature."
    ),
    "is_goal": (
        "This IS the Track A binary target. "
        "Using it as a feature is the definition of data leakage."
    ),
}

CATEGORICAL_COLS = [
    "shot_outcome",
    "shot_body_part",
    "shot_technique",
    "shot_type",
    "play_pattern",
]


# ---------------------------------------------------------------------------
# Source detection (StatsBomb vs Wyscout)
# ---------------------------------------------------------------------------

def detect_sources(df: pd.DataFrame) -> dict:
    """
    Wyscout shots: statsbomb_xg is NaN, shot_type == 'Shot', body_part == 'Unknown'.
    StatsBomb shots: statsbomb_xg is not NaN, granular categories.
    """
    wyscout_mask = df["statsbomb_xg"].isna()
    statsbomb_mask = ~wyscout_mask
    return {
        "statsbomb_rows": int(statsbomb_mask.sum()),
        "wyscout_rows": int(wyscout_mask.sum()),
        "statsbomb_goal_rate": round(float(df.loc[statsbomb_mask, "is_goal"].mean()) * 100, 2),
        "wyscout_goal_rate": round(float(df.loc[wyscout_mask, "is_goal"].mean()) * 100, 2),
    }


def build_value_mapping(df: pd.DataFrame) -> dict[str, list]:
    """Canonical vocabulary for all downstream phases."""
    return {
        col: sorted(df[col].dropna().unique().tolist())
        for col in CATEGORICAL_COLS
        if col in df.columns
    }


# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------

def run_audit(df: pd.DataFrame) -> dict:
    result: dict = {}

    result["n_rows"] = len(df)
    result["n_cols"] = len(df.columns)
    result["columns"] = list(df.columns)
    result["dtypes"] = {c: str(df[c].dtype) for c in df.columns}

    result["n_duplicates"] = int(df.duplicated().sum())
    result["n_unique_shots"] = int(df["shot_id"].nunique()) if "shot_id" in df.columns else None
    result["n_unique_matches"] = int(df["match_id"].nunique()) if "match_id" in df.columns else None

    null_counts = df.isnull().sum()
    result["null_counts"] = {c: int(n) for c, n in null_counts.items() if n > 0}
    result["null_pct"] = {c: round(n / len(df) * 100, 2) for c, n in null_counts.items() if n > 0}

    result["n_goals"] = int(df["is_goal"].sum())
    result["n_non_goals"] = len(df) - result["n_goals"]
    result["goal_rate_pct"] = round(result["n_goals"] / len(df) * 100, 2)

    for col in CATEGORICAL_COLS + ["is_first_time"]:
        if col in df.columns:
            result[f"dist_{col}"] = df[col].value_counts().to_dict()

    for col in ["season_id", "competition_id"]:
        if col in df.columns:
            result[f"dist_{col}"] = df[col].value_counts().to_dict()

    for col in ["location_x", "location_y", "distance_to_goal", "shot_angle"]:
        if col in df.columns:
            result[f"range_{col}"] = {
                "min": round(float(df[col].min()), 3),
                "max": round(float(df[col].max()), 3),
                "mean": round(float(df[col].mean()), 3),
            }

    result["column_roles"] = {c: COLUMN_ROLES.get(c, "unknown") for c in df.columns}
    result["forbidden_present"] = {c: r for c, r in FORBIDDEN_REASONS.items() if c in df.columns}
    result["forbidden_absent"] = [c for c in FORBIDDEN_REASONS if c not in df.columns]

    foot_cols = [c for c in df.columns if "foot" in c.lower() or "preferred" in c.lower()]
    result["preferred_foot_cols"] = foot_cols
    result["preferred_foot_available"] = len(foot_cols) > 0

    result["sources"] = detect_sources(df)
    result["value_mapping"] = build_value_mapping(df)

    return result


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_report(result: dict) -> str:
    lines = ["# Phase 1 -- Data Audit Report\n"]

    lines.append("## 1. Dataset Shape\n")
    lines.append(f"- Total rows: **{result['n_rows']:,}**")
    lines.append(f"- Columns: **{result['n_cols']}**")
    lines.append(f"- Unique shots: **{result.get('n_unique_shots', '?'):,}**")
    lines.append(f"- Unique matches: **{result.get('n_unique_matches', '?'):,}**")
    lines.append(f"- Duplicate rows: **{result['n_duplicates']}**\n")

    lines.append("## 2. Data Sources\n")
    src = result["sources"]
    lines.append("This dataset combines StatsBomb and Wyscout open data.\n")
    lines.append("| Source | Rows | Goal rate |")
    lines.append("|--------|------|-----------|")
    lines.append(f"| StatsBomb | {src['statsbomb_rows']:,} | {src['statsbomb_goal_rate']}% |")
    lines.append(f"| Wyscout | {src['wyscout_rows']:,} | {src['wyscout_goal_rate']}% |")
    lines.append(
        "\nWyscout shots are identified by `statsbomb_xg = NaN` and `shot_type = 'Shot'`."
        " They lack granular technique/body-part categories."
        " Phase 2 will decide how to handle them.\n"
    )

    lines.append("## 3. Class Balance (Track A Target)\n")
    lines.append("- Target: `is_goal` (bool)")
    lines.append(f"- Goals: **{result['n_goals']:,}** ({result['goal_rate_pct']}%)")
    lines.append(f"- Non-goals: **{result['n_non_goals']:,}** ({100 - result['goal_rate_pct']}%)")
    lines.append(
        "\n> ~10% goal rate is typical. Class imbalance will be addressed in Phase 5/6.\n"
    )

    lines.append("## 4. Column Role Classification\n")
    roles: dict[str, list[str]] = {}
    for col, role in result["column_roles"].items():
        roles.setdefault(role, []).append(col)

    role_descriptions = {
        "identifier":        "Unique IDs -- for joining only, not modeling",
        "match_metadata":    "Match context -- for splitting and grouping",
        "candidate_feature": "Available at shot release -- usable as model inputs",
        "target_track_a":    "Track A target -- never a feature",
        "target_track_b":    "Track B target -- never a Track A feature",
        "target_derived":    "Derived from outcome -- never a feature",
        "forbidden_post_shot": "POST-SHOT: reveals ball position after release -- FORBIDDEN",
        "unknown":           "Unclassified",
    }
    for role, cols in sorted(roles.items()):
        desc = role_descriptions.get(role, "")
        lines.append(f"**{role}** -- *{desc}*")
        for c in cols:
            lines.append(f"  - `{c}`")
        lines.append("")

    lines.append("## 5. Leakage Analysis\n")
    lines.append(
        "Leakage boundary: *'Was this information observable at the exact moment "
        "the player's foot made contact with the ball?'*\n"
    )
    lines.append("### Forbidden columns present in dataset:\n")
    for col, reason in result["forbidden_present"].items():
        lines.append(f"**`{col}`** -- {reason}\n")

    lines.append("## 6. Shot Type Distribution\n")
    if "dist_shot_type" in result:
        lines.append("| Shot type | Count |")
        lines.append("|-----------|-------|")
        for k, v in sorted(result["dist_shot_type"].items(), key=lambda x: -x[1]):
            lines.append(f"| {k} | {v:,} |")
    lines.append(
        "\n> **Phase 2 decision:** Exclude penalties (different physics, near-deterministic)."
        " Free kicks are flagged for potential separate treatment.\n"
    )

    lines.append("## 7. Categorical Distributions\n")
    for col in ["shot_body_part", "shot_technique", "play_pattern"]:
        key = f"dist_{col}"
        if key in result:
            lines.append(f"**{col}**")
            for k, v in sorted(result[key].items(), key=lambda x: -x[1]):
                lines.append(f"  - {k}: {v:,}")
            lines.append("")

    lines.append("## 8. Spatial Feature Ranges\n")
    for col in ["location_x", "location_y", "distance_to_goal", "shot_angle"]:
        key = f"range_{col}"
        if key in result:
            r = result[key]
            lines.append(f"- `{col}`: [{r['min']}, {r['max']}], mean={r['mean']}")
    lines.append("")

    lines.append("## 9. Null Summary\n")
    if result["null_counts"]:
        lines.append("| Column | Null count | Null % | Note |")
        lines.append("|--------|-----------|--------|------|")
        notes = {"statsbomb_xg": "Wyscout shots -- expected, no StatsBomb xG for these"}
        for col, cnt in sorted(result["null_counts"].items(), key=lambda x: -x[1]):
            pct = result["null_pct"].get(col, 0)
            note = notes.get(col, "")
            lines.append(f"| `{col}` | {cnt:,} | {pct}% | {note} |")
    else:
        lines.append("No unexpected nulls.\n")
    lines.append("")

    lines.append("## 10. Preferred Foot Metadata\n")
    if result["preferred_foot_available"]:
        lines.append(f"Found: {result['preferred_foot_cols']}\n")
    else:
        lines.append(
            "**Not in raw data.** Phase 2 will infer preferred foot statistically: "
            "the foot used on >70% of a player's non-header shots is treated as preferred. "
            "This enables `is_weak_foot` = 1 when the shot foot differs from preferred foot.\n"
        )

    lines.append("## 11. Value Mapping Dictionary\n")
    lines.append("Canonical vocabulary for all downstream phases:\n")
    lines.append("```json")
    lines.append(json.dumps(result["value_mapping"], indent=2, ensure_ascii=False))
    lines.append("```\n")

    lines.append("## 12. Proposed Modeling Sample\n")
    lines.append(
        "**Track A (Real xG)**\n"
        "- Population: all shots where `shot_type != 'Penalty'`\n"
        "- Target: `is_goal` (bool -> int)\n"
        "- Candidate features: `location_x`, `location_y`, `distance_to_goal`, `shot_angle`,\n"
        "  `shot_body_part`, `shot_technique`, `shot_type`, `play_pattern`, `is_first_time`, `minute`\n"
        "- Forbidden: `end_location_x`, `end_location_y`, `shot_outcome`, `statsbomb_xg`, `is_goal`\n\n"
        "**Track B (Teacher Imitation)**\n"
        "- Population: StatsBomb shots only (`statsbomb_xg` is not null) excluding penalties\n"
        "- Target: `statsbomb_xg` (continuous regression)\n"
        "- Same candidate features as Track A\n"
        "- Forbidden: `is_goal`, `shot_outcome`, `end_location_x`, `end_location_y`\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 1 -- Data Audit & Problem Definition")
    print("=" * 60)

    df = load_shot_data()
    print(f"\nDataset: {len(df):,} rows x {len(df.columns)} columns")

    result = run_audit(df)

    print(f"\n{'--'*20}")
    print(f"Goal rate       : {result['goal_rate_pct']}%  "
          f"({result['n_goals']:,} goals / {result['n_rows']:,} shots)")
    print(f"Unique matches  : {result.get('n_unique_matches', '?'):,}")
    print(f"Duplicates      : {result['n_duplicates']}")
    print(f"Preferred foot  : {'PRESENT' if result['preferred_foot_available'] else 'NOT FOUND -- infer in Phase 2'}")

    src = result["sources"]
    print(f"\nStatsBomb shots : {src['statsbomb_rows']:,} (goal rate {src['statsbomb_goal_rate']}%)")
    print(f"Wyscout shots   : {src['wyscout_rows']:,} (goal rate {src['wyscout_goal_rate']}%)")

    print(f"\n{'--'*20}")
    print("Leakage -- FORBIDDEN columns present in dataset:")
    for col in result["forbidden_present"]:
        print(f"  [FORBIDDEN] {col}")

    print(f"\n{'--'*20}")
    print("Shot type distribution:")
    for k, v in sorted(result.get("dist_shot_type", {}).items(), key=lambda x: -x[1]):
        print(f"  {k:<20}: {v:,}")

    print(f"\n{'--'*20}")
    print("Spatial feature ranges:")
    for col in ["distance_to_goal", "shot_angle", "location_x", "location_y"]:
        key = f"range_{col}"
        if key in result:
            r = result[key]
            print(f"  {col:<20}: [{r['min']}, {r['max']}]  mean={r['mean']}")

    report_md = generate_report(result)
    report_path = _dir("outputs.reports_dir") / "01_audit_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"\nReport saved    : {report_path}")

    mapping_path = _dir("outputs.tables_dir") / "01_value_mapping.json"
    mapping_path.write_text(
        json.dumps(result["value_mapping"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Value mapping   : {mapping_path}")

    null_rows = [
        {"column": c, "null_count": n, "null_pct": result["null_pct"].get(c, 0)}
        for c, n in result["null_counts"].items()
    ]
    if null_rows:
        null_df = pd.DataFrame(null_rows).sort_values("null_count", ascending=False)
        null_path = _dir("outputs.tables_dir") / "01_null_summary.csv"
        null_df.to_csv(null_path, index=False)
        print(f"Null summary    : {null_path}")

    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
