---
name: opta-events-reference
description: >
  Complete reference for Opta football event data: JSONP file format, every
  event typeId and its meaning, every qualifier ID and when it appears, the
  0-100 → real-pitch coordinate conversion, the parallel-array structure of
  the team-setup event (typeId 34), synthesised carry events (typeId -1),
  and the sequence columns populated in a second pass.

  Use this skill whenever working on the events pipeline (parser.py, carries.py,
  xt.py, processor.py, db.py), writing new event-level SQL or aggregations,
  designing the possession-sequence classifier, building new gold-layer tables
  from silver.events, or answering any question that touches raw event JSON,
  event types, or qualifier IDs.
---

# Opta Events Reference

## 1. File Format — JSONP Wrapper

Every raw file starts with a hash prefix before the JSON payload:

```
W3c016d8247...({"matchInfo": {...}, "liveData": {...}})
```

**Strip strategy** (used in all loaders):
```python
raw = path.read_text()
start = raw.index("{")
payload = raw[start:].rstrip()
if payload.endswith(")"):
    payload = payload[:-1]
data = json.loads(payload)
```

Top-level keys: `matchInfo`, `liveData`

---

## 2. Top-Level Structure

```json
{
  "matchInfo": {
    "id":           "source_match_id",
    "stage":        {"id": "source_stage_id"},
    "week":         "matchday_number",
    "localDate":    "YYYY-MM-DD",
    "localTime":    "HH:MM:SS",
    "contestant":   [{"id": "...", "name": "...", "position": "home|away"}, ...],
    "venue":        {"id": "...", "longName": "...", "latitude": "...", "longitude": "..."},
    "var":          "1|0",
    "coverageLevel":"...",
    "lastUpdated":  "ISO8601Z"
  },
  "liveData": {
    "matchDetails": {
      "winner":   "home|away|draw",
      "scores":   {"ft": {"home": N, "away": N}, "ht": {"home": N, "away": N}},
      "period":   [{"id": 1, "announcedInjuryTime": N}, {"id": 2, ...}],
      "matchLengthMin": N,
      "matchLengthSec": N
    },
    "event": [ ... ]   // ← the event stream
  }
}
```

---

## 3. Event Object Structure

```json
{
  "id":           "uuid_string",       // source_event_id (The unique id for this event within Opta’s entire database of all events in all games)
  "eventId":      1,                   // provider_event_id (NOT sequential The unique id for this event within this game for each team – used as a reference for qualifier_id values)
  "typeId":       34,                  // event classification
  "periodId":     1,                   // 1=H1, 2=H2, 3=ET1, 4=ET2, 16=pre-match
  "timeMin":      0,
  "timeSec":      0,
  "contestantId": "team_source_id",
  "playerId":     "player_source_id",  // absent for team-level events
  "playerName":   "string",
  "x":            0.0,                 // 0-100 scale
  "y":            0.0,                 // 0-100 scale
  "outcome":      1,                   // 1=success, 0=failure
  "timeStamp":    "ISO8601Z",
  "qualifier":    [ {"id": N, "qualifierId": N, "value": "..."}, ... ]
}
```

---

## 4. Coordinate System

Provider uses a **0-100 scale** for both axes. Convert to real pitch dimensions:

```python
real_x = (provider_x / 100) * 105   # metres, 0 = own goal line
real_y = (provider_y / 100) * 68    # metres
```

- **x = 0**: own goal line (attacking left → right convention)
- **x = 100 (→ 105m)**: opponent's goal line
- **y = 0**: bottom touchline, **y = 100 (→ 68m)**: top touchline

Qualifier coordinates (pass end, blocked, goal mouth) also arrive on 0-100 and must be scaled the same way. Goal-mouth Z (height) is also 0-100 but has no fixed real-world dimension — scale is kept as-is or documented separately.

---

## 5. Key Event typeIds

### Critical pipeline events

| typeId | Name | Notes |
|--------|------|-------|
| **34** | Team set up | One per team per match. Contains full squad via parallel-array qualifiers (see §6). `periodId=16` (pre-match). |
| **18** | Player off | Substitution exit. `playerId` = outgoing player. `timeMin` = minute_out. |
| **19** | Player on | Substitution entry. `playerId` = incoming player. `timeMin` = minute_in. Q145 = new formation slot. |
| **17** | Card | Q33 = straight red, Q32 = second yellow, Q31 = yellow (yellow-only has no lineup impact). |
| **30** | End | End of period. |
| **32** | Start | Start of period. Q127 = direction of play. |
| **43** | Deleted event | Event was removed post-match. Keep `typeId=43` in DB; do not process as original type. |

### Match-action events (partial list — see §5.1 for full table)

| typeId | Name | Outcome 1 | Outcome 0 |
|--------|------|-----------|-----------|
| 1  | Pass | Completed | Incomplete |
| 2  | Offside Pass | — | — |
| 3  | Take On | Dribble won | Dribble lost |
| 4  | Foul | (foul committed) | — |
| 7  | Tackle | Ball won+retained | Ball won, possession lost |
| 8  | Interception | Intercepted | — |
| 10 | Save | Saved | — |
| 11 | Claim | Caught | — |
| 12 | Clearance | — | — |
| 13 | Miss | — | — |
| 14 | Post | — | — |
| 15 | Attempt Saved | — | — |
| 16 | Goal | — | — |
| 44 | Aerial | Won | Lost |
| 45 | Challenge | (opponent dribbled past) | — |
| 49 | Ball recovery | Won | — |
| 50 | Dispossessed | (lost ball) | — |
| 52 | Keeper pick-up | — | — |

### Synthesised events (not from provider)

| typeId | Name | Notes |
|--------|------|-------|
| **-1** | Carry | Created by `carries.py`. No `source_event_id`, no `provider_event_id`, no `raw_data`. `outcome = 'success'`. Inserted between consecutive events where the ball moved without a recorded action. |

---

### 5.1 Full Event Type Table

See the `optaEventCodes` JS variable in `data/mapping/opta-events.js` for all ~70 type IDs. The most analytically significant ones beyond §5:

| typeId | Name |
|--------|------|
| 5  | Out (throw-in / goal kick) |
| 6  | Corner Awarded |
| 9  | Turnover (deprecated, no longer used) |
| 20 | Player retired (injured, no subs left) |
| 35 | Player changed position |
| 40 | Formation change |
| 41 | Punch (GK) |
| 42 | Good Skill (deprecated) |
| 49 | Ball recovery |
| 51 | Error (leads to shot/goal) |
| 54 | Smother (GK) |
| 59 | Keeper Sweeper |
| 61 | Ball touch |
| 66 | Possession Data (every 5 mins) |
| 70 | Injury Time Announcement |
| 74 | Blocked Pass |

---

## 6. typeId 34 — Team Set Up (Parallel Arrays)

This event holds the full 23-player squad as **comma-separated parallel arrays** in qualifiers. All arrays are the same length and index-aligned.

```
Q30  = "pid1, pid2, pid3, ..."          // source_player_ids
Q44  = "1, 2, 2, 3, 2, ..."             // position codes
Q59  = "1, 7, 5, 8, ..."                // shirt numbers
Q131 = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, ..."  // formation slots
Q194 = "pid_of_captain"                 // single value (captain's source_player_id)
Q130 = "formation_raw_id"               // single value (e.g. "13" for 4-3-3)
Q227 = "0, 0, 0, ..."                   // unknown status flag (all zeros observed)
Q197 = "kit_id"                         // kit identifier
```

**Position code mapping (Q44)**:
| Code | Label |
|------|-------|
| 1 | GK |
| 2 | DEF |
| 3 | MID |
| 4 | FWD |
| 5 | SUB |

**Formation slot (Q131)**:
- Values 1–11: starting XI (position in formation)
- Value 0: bench (not in starting XI)

`starting_xi = (formation_slot > 0)`
`minute_in = 0` for starters, `NULL` for bench (updated to actual minute via typeId 19)

---

## 7. Key Qualifier IDs

### Coordinate qualifiers (value is 0-100, must be scaled)

| qualifierId | Name | Axis | Column in silver.events |
|-------------|------|------|------------------------|
| 140 | Pass End X | x | `end_x` |
| 141 | Pass End Y | y | `end_y` |
| 146 | Blocked x co-ordinate | x | `blocked_x` |
| 147 | Blocked y co-ordinate | y | `blocked_y` |
| 102 | Goal mouth y co-ordinate | y | `goal_mouth_y` |
| 103 | Goal mouth z co-ordinate | z | `goal_mouth_z` |
| 230 | GK X Coordinate | x | (not in silver schema) |
| 231 | GK Y Coordinate | y | (not in silver schema) |

### Lineup / squad qualifiers (typeId 34)

| qualifierId | Name | Value format |
|-------------|------|-------------|
| 30 | Involved / Player IDs | Comma-separated source_player_ids |
| 44 | Player position | Comma-separated codes (1-5) |
| 59 | Jersey number | Comma-separated integers |
| 130 | Team formation | Single raw formation ID |
| 131 | Team player formation | Comma-separated slots 1-11 (0=bench) |
| 194 | Captain | Single source_player_id |
| 227 | Status flag | Comma-separated integers (meaning unknown) |
| 197 | Team kit | Kit ID |

### Substitution qualifiers

| qualifierId | Name | Event | Notes |
|-------------|------|-------|-------|
| 145 | Formation slot | typeId 19 | New formation position of the sub entering |
| 293 | Sub flag | typeId 19 | Unknown, preserved as `raw_sub_flag` |
| 41 | Injury | typeId 18 | Sub caused by injury |
| 42 | Tactical | typeId 18 | Tactical substitution |

### Card qualifiers (typeId 17)

| qualifierId | Name | Lineup impact |
|-------------|------|--------------|
| 31 | Yellow Card | None |
| 32 | Second yellow | `exit_reason = 'second_yellow'`, sets `minute_out` |
| 33 | Red card | `exit_reason = 'red_card'`, sets `minute_out` |

### Shot / pass qualifiers (selection)

| qualifierId | Name |
|-------------|------|
| 1 | Long ball |
| 2 | Cross |
| 3 | Head pass |
| 4 | Through ball |
| 5 | Free kick taken |
| 6 | Corner taken |
| 9 | Penalty |
| 15 | Head (body part) |
| 20 | Right footed |
| 72 | Left footed |
| 210 | Assist |
| 214 | Big Chance |
| 55 | Related event ID (assist link) |

Full qualifier list: see `data/mapping/opta-qualifiers.js`.

---

## 8. Carry Detection Rules (`carries.py`)

Carries are **synthesised** between consecutive provider events when the ball moved without a recorded event. Key rules (implemented in `carries.py`):

**Never insert a carry when**:
- Next event is `BallTouch`, `BallRecovery`, `Aerial`, or `CornerAwarded`
- Current event is `Foul` or `Card`
- Current = `MissedShot` AND next = `BallTouch`

**Insert carry when** (same team, coordinate mismatch, valid coords on both sides):
- Current = `Pass` (success) → next = `Pass | Shot | MissedShot | SavedShot | Dispossessed | Foul`
- Current = `BallRecovery | KeeperPickup | Interception | Claim` → next = `Pass | Shot | MissedShot | SavedShot`
- Current = `Clearance` → next = `Pass`
- Current = `Tackle` → next = `Pass` (same player, same team)

Coordinate mismatch threshold: `abs(cur.end_x - nxt.x) > 0.01` OR `abs(cur.end_y - nxt.y) > 0.01`

Carry row fields: `event_type='Carry'`, `type_id=None`, `source_event_id=None`, `outcome='success'`, `raw_data=None`.

---

## 9. silver.events Column Reference

```sql
source_event_id       -- provider's UUID (null for carries)
provider_event_id     -- provider's sequential eventId (null for carries)
match_id              -- FK → silver.matches
period                -- 1|2|3|4
minute, second        -- match clock
timestamp             -- ISO8601 UTC
team_id               -- FK → silver.teams (resolved from source_team_id)
source_team_id        -- provider's contestantId (kept for cross-row joins)
player_id             -- FK → silver.players (nullable)
source_player_id      -- provider's playerId (kept for cross-row joins)
player_name           -- denormalised from event JSON
jersey_number         -- from typeId 34 Q59 lookup
team_name             -- denormalised
opposition_team_name  -- denormalised
h_a                   -- 'home'|'away'
type_id               -- raw provider typeId (-1 for carries)
event_type            -- human label from opta-events.js mapping
outcome               -- 'success'|'failure'|null
x, y                  -- real pitch coords (metres, scaled from 0-100)
end_x, end_y          -- destination coords (fallback: x/y if not set)
blocked_x, blocked_y  -- where shot was blocked
goal_mouth_z, goal_mouth_y -- goalmouth coordinates
start_zone_value_xt   -- xT at (x, y); null for non-Pass/Carry
end_zone_value_xt     -- xT at (end_x, end_y)
xt                    -- end_zone - start_zone
sequence_id           -- possession sequence ID (populated in second pass)
sequence_start        -- bool: first event of a sequence
sequence_end          -- bool: last event of a sequence
sequence_event_number -- position within sequence
raw_data              -- full provider JSON payload (null for carries)
```

---

## 10. xT Grid

8 rows (Y, 0-68m) × 12 columns (X, 0-105m). Calculated for `Pass` and `Carry` only.

```python
col = int(np.clip(x / 105 * 12, 0, 11))
row = int(np.clip(y / 68  *  8, 0,  7))
xt_value = XT_GRID[row, col]
```

Grid is defined in `src/silver/events/xt.py`. The grid is symmetric top/bottom (rows 0 and 7 equal, rows 1 and 6 equal, etc.).

---

## 11. Pipeline Processing Order

```
1. Extract source_match_id (cheap — index into first '{')
2. Resolve match_id from silver.matches (skip if not found)
3. Idempotency check: skip if silver.events already has rows for match_id
4. parse_event_file() → list[dict]
   - build_team_mapping() from matchInfo.contestant
   - build_jersey_mapping() from typeId:34 events
   - _extract_event() per event (skips typeId:43 Deleted, skips 'Unknown')
5. calculate_carries() → insert synthesised carry rows
6. calculate_xt() → add xT columns for Pass/Carry
7. resolve_fk_columns() → map source IDs → internal IDs
8. insert_events() → silver.events (page_size=500)
9. conn.commit()
```

---

## 12. Sequence Columns (populated separately)

`sequence_id`, `sequence_start`, `sequence_end`, `sequence_event_number` are **NULL on initial load** and populated in a second pass by a possession-sequence classifier (not yet implemented as of schema definition).

A **possession sequence** = a continuous chain of events by the same team without a change of possession. Typical definition:
- Starts after: opponent event, ball-out-of-play, goalkeeper recovery, or start of period
- Ends at: loss of possession, foul conceded, ball out of play, end of period
- Carries are part of the same sequence as the surrounding events

In the references folder, you have the raw opta-event.js and opta-qualifiers.js