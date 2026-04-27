---
name: spadl-mapping
description: >
  Complete reference for the SPADL (Soccer Player Action Description Language) mapping
  layer applied to silver.events Opta data. Covers the three SPADL columns
  (spadl_type_id, spadl_result_id, spadl_bodypart_id) added to silver.events, the full
  Opta event_type → SPADL action type mapping, qualifier-based sub-type detection,
  result and body-part mapping, the always-success / always-fail override sets, the
  backfill function, and the VAEP/xT downstream contract.

  Use this skill whenever working on spadl.py, writing queries that filter or group
  by spadl_type_id / spadl_result_id / spadl_bodypart_id, building gold-layer VAEP
  features, or extending the SPADL action set.
---

# SPADL Mapping Reference

## 1. What Is SPADL and Why It Lives in silver.events

SPADL normalises vendor-specific event formats into a fixed schema. In this project, three
SPADL columns are added **directly to `silver.events`** as computed attributes rather than
in a separate table, so the raw Opta event and the SPADL annotation always travel together.

Events that don't map to any SPADL action get `NULL` in all three columns.

**Module**: `src/silver/events/spadl.py`  
**Entry point**: `calculate_spadl(df)` — pure transform, no DB access.  
**Backfill CLI**: `python -m src.silver.events.spadl [--match-ids N …] [--limit N]`

---

## 2. Three New Columns on silver.events

| Column | Type | Description |
|--------|------|-------------|
| `spadl_type_id` | `Int16` (nullable) | SPADL action type (0–22). NULL = non-action. |
| `spadl_result_id` | `Int16` (nullable) | SPADL result (0–5). NULL = non-action. |
| `spadl_bodypart_id` | `Int16` (nullable) | SPADL body part (0–5). NULL = non-action. |

All three are `NULL` together or populated together — never partial.

---

## 3. SPADL Action Types (spadl_type_id, 0–22)

| type_id | Constant | SPADL Name | Notes |
|---------|----------|------------|-------|
| 0 | `PASS` | pass | Default pass in open play |
| 1 | `CROSS` | cross | Q2 present on a Pass event |
| 2 | `THROW_IN` | throw_in | Q107 present on a Pass event |
| 3 | `FREEKICK_CROSSED` | freekick_crossed | Q5 + Q2 on a Pass event |
| 4 | `FREEKICK_SHORT` | freekick_short | Q5 (no Q2) on a Pass event |
| 5 | `CORNER_CROSSED` | corner_crossed | Q6 + Q2 on a Pass event |
| 6 | `CORNER_SHORT` | corner_short | Q6 (no Q2) on a Pass event |
| 7 | `TAKE_ON` | take_on | Opta "Take On" event |
| 8 | `FOUL` | foul | Opta "Foul" event |
| 9 | `TACKLE` | tackle | Opta "Tackle" event |
| 10 | `INTERCEPTION` | interception | Opta "Interception" or "Blocked Pass" |
| 11 | `SHOT` | shot | Opta shot events (Miss/Post/Attempt Saved/Goal) |
| 12 | `SHOT_PENALTY` | shot_penalty | Shot + Q9 (penalty) |
| 13 | `SHOT_FREEKICK` | shot_freekick | Shot + Q26 (direct free kick) |
| 14 | `KEEPER_SAVE` | keeper_save | Opta "Save" (excluding Q94 team-save) |
| 15 | `KEEPER_CLAIM` | keeper_claim | Opta "Claim" |
| 16 | `KEEPER_PUNCH` | keeper_punch | Opta "Punch" |
| 17 | `KEEPER_PICK_UP` | keeper_pick_up | Opta "Keeper pick-up" |
| 18 | `CLEARANCE` | clearance | Opta "Clearance" |
| 19 | `BAD_TOUCH` | bad_touch | Opta "Ball touch" |
| 20 | *(non_action)* | non_action | Internal only — all three columns = NULL |
| 21 | `DRIBBLE` | dribble | Opta "Carry" (synthesised, type_id = -1) |
| 22 | `GOALKICK` | goalkick | Q124 on a Pass event |

---

## 4. SPADL Results (spadl_result_id, 0–5)

| result_id | Constant | Name | When assigned |
|-----------|----------|------|---------------|
| 0 | `FAIL` | fail | Default failure; always-fail types; fouls; bad touches |
| 1 | `SUCCESS` | success | Default success; always-success types; goals |
| 2 | `OFFSIDE` | offside | event_type = "Offside Pass" |
| 3 | `OWNGOAL` | owngoal | Goal + Q28 (own goal qualifier) |
| 4 | `YELLOW_CARD` | yellow_card | *(reserved; not yet populated from foul+card logic)* |
| 5 | `RED_CARD` | red_card | *(reserved; not yet populated)* |

### Always-override sets

These types ignore the Opta `outcome` field entirely:

```python
_ALWAYS_SUCCESS = {KEEPER_SAVE, KEEPER_PUNCH, KEEPER_PICK_UP, CLEARANCE, DRIBBLE}
_ALWAYS_FAIL    = {FOUL, BAD_TOUCH}
```

### Result for shots

Shots use `outcome` unless the event_type is "Goal":
- `event_type == "Goal"` + Q28 → `OWNGOAL`
- `event_type == "Goal"` (no Q28) → `SUCCESS`
- `outcome == "success"` → `SUCCESS`
- anything else → `FAIL`

---

## 5. SPADL Body Parts (spadl_bodypart_id, 0–5)

| bodypart_id | Constant | Name | Qualifier trigger |
|-------------|----------|------|-------------------|
| 0 | `FOOT` | foot | Default (no foot/head/other qualifier) |
| 1 | `HEAD` | head | Q15, Q3, or Q168 present |
| 2 | `OTHER` | other | Q21 present |
| 3 | *(unused)* | head/other | Wyscout only — not assigned here |
| 4 | `FOOT_LEFT` | foot_left | Q72 present |
| 5 | `FOOT_RIGHT` | foot_right | Q20 present |

Dribbles (`DRIBBLE`) always get `FOOT` regardless of qualifiers.

---

## 6. Opta event_type → SPADL type_id Decision Tree

```
event_type == "Pass" or "Offside Pass"
    "Offside Pass"  → PASS  (result = OFFSIDE)
    Q6 + Q2         → CORNER_CROSSED
    Q6              → CORNER_SHORT
    Q5 + Q2         → FREEKICK_CROSSED
    Q5              → FREEKICK_SHORT
    Q107            → THROW_IN
    Q124            → GOALKICK
    Q2              → CROSS
    (none)          → PASS

event_type == "Take On"           → TAKE_ON
event_type == "Foul"              → FOUL
event_type == "Tackle"            → TACKLE
event_type in ("Interception",
               "Blocked Pass")    → INTERCEPTION

event_type in ("Miss", "Post",
               "Attempt Saved",
               "Goal")
    Q9              → SHOT_PENALTY
    Q26             → SHOT_FREEKICK
    (none)          → SHOT

event_type == "Save"
    Q94             → NULL (non-action — team save, not GK save)
    (none)          → KEEPER_SAVE

event_type == "Claim"             → KEEPER_CLAIM
event_type == "Punch"             → KEEPER_PUNCH
event_type == "Keeper pick-up"    → KEEPER_PICK_UP
event_type == "Clearance"         → CLEARANCE
event_type == "Ball touch"        → BAD_TOUCH
event_type == "Carry" / type_id=-1 → DRIBBLE

everything else                   → NULL (non-action)
```

---

## 7. Qualifier IDs Used by SPADL Mapping

| Qualifier ID | Meaning | Used for |
|-------------|---------|----------|
| 2 | Cross | Distinguish cross/corner_crossed/freekick_crossed from pass |
| 3 | Head pass | Body part = head |
| 5 | Free kick taken | Free kick pass sub-types |
| 6 | Corner taken | Corner pass sub-types |
| 9 | Penalty | Shot → SHOT_PENALTY |
| 15 | Head (body part) | Body part = head |
| 20 | Right foot | Body part = foot_right |
| 21 | Other body part | Body part = other |
| 26 | Direct free kick | Shot → SHOT_FREEKICK |
| 28 | Own goal | Goal result → OWNGOAL |
| 72 | Left foot | Body part = foot_left |
| 94 | Team save (not GK) | Save → NULL (non-action) |
| 107 | Throw-in | Pass → THROW_IN |
| 124 | Goal kick | Pass → GOALKICK |
| 168 | Head (body part, alt) | Body part = head |

---

## 8. Public API

### `calculate_spadl(df) → DataFrame`

Pure transform. Expects a DataFrame with these columns from `silver.events`:

| Input column | Used for |
|---|---|
| `event_type` | Primary dispatch key |
| `outcome` | Result mapping (success/failure string) |
| `type_id` | Carry detection (type_id == -1) |
| `raw_data` | Qualifier extraction (JSON string or dict) |

Returns the same DataFrame with three new columns appended:
`spadl_type_id`, `spadl_result_id`, `spadl_bodypart_id` (all `pd.Int16Dtype()`).

### `_get_qualifier_ids(raw_data) → Set[int]`

Extracts qualifier IDs from a `raw_data` payload. Handles `None`, `NaN`, JSON string,
and dict. Returns an empty set on parse failure. Looks for `qualifierId` then falls back
to `id` in each qualifier object.

### `backfill_spadl(conn, match_ids=None, limit=None, batch_size=50) → dict`

Reads events from `silver.events`, applies `calculate_spadl`, then bulk-updates the three
columns using `psycopg2.extras.execute_values`. Auto-discovers unprocessed matches (those
where no events have `spadl_type_id IS NOT NULL`) when `match_ids` is not given.

Returns `{"matches_processed": N, "events_updated": N}`.

---

## 9. Downstream Contract for VAEP / xT

### Which events are SPADL actions?

```sql
WHERE spadl_type_id IS NOT NULL
```

### Action type groups for gold-layer aggregation

```sql
-- Move actions (used for xT valuation)
WHERE spadl_type_id IN (0,1,2,3,4,5,6,7,10,17,18,21,22)  -- pass family + dribble + clearance + goalkick

-- Shot actions
WHERE spadl_type_id IN (11, 12, 13)

-- Defensive actions
WHERE spadl_type_id IN (8, 9, 10)  -- foul, tackle, interception

-- GK actions
WHERE spadl_type_id IN (14, 15, 16, 17)
```

### VAEP game-state window

VAEP requires the **last 3 SPADL actions** per period per match, ordered by
`(match_id, period, minute, second)`. Filter `WHERE spadl_type_id IS NOT NULL` before
building the sliding window.

### Direction normalisation

For VAEP/xT, all coordinates must be oriented so the acting team attacks toward x=105.
Apply the flip for the team not playing left-to-right:
```
start_x = 105 - start_x    end_x = 105 - end_x
start_y = 68  - start_y    end_y = 68  - end_y
```
The `silver.events` table stores raw Opta coordinates (attacking direction varies by
period and team). Check `h_a` and period for the correct flip.

---

## 10. Synthetic Carries as Dribbles

Synthesised carry events (created by `carries.py`, stored with `type_id = -1`) map to
SPADL `DRIBBLE` (type_id = 21):
- `spadl_type_id = 21` always
- `spadl_result_id = 1` (SUCCESS) always
- `spadl_bodypart_id = 0` (FOOT) always

These are the SPADL equivalent of the `socceraction` library's synthetic dribble insertion.
No re-synthesis is needed — the carries already satisfy the same distance/time contract.

---

## 11. Known Limitations and Future Work

| Issue | Status |
|-------|--------|
| Yellow/red card results on fouls (Q31/Q32/Q33) | Not yet wired — fouls always get `FAIL` |
| Keeper claim result (success/fail based on outcome) | Mapped via `outcome` — correct |
| Own-goal end coordinates (should be opponent's goal at 105, 34) | Not yet adjusted in spadl.py |
| Atomic-SPADL conversion | Not implemented — see SPADL.md §7 |
| VAEP feature builder | Not implemented — see SPADL.md §9 |
