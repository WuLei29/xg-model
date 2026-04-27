---
name: football-silver-schema
description: >
  Complete reference for the football analytics relational schema (silver layer):
  all 10 tables with columns, constraints and FK relationships, the medallion
  architecture (bronze/silver/gold), idempotency and conflict patterns, the
  5-pass match_lineups ingestion strategy, the squad diff strategy, source-ID
  resolution via SQL JOIN, and the temporal join pattern for player_squads.

  Use this skill whenever writing or debugging ingestion scripts (load_matches.py,
  load_match_lineups.py, squad_processor.py, load_teams.py), writing silver-layer
  SQL queries, designing new gold-layer tables, reasoning about FK resolution,
  or any question about the data model, table relationships, or ingestion patterns.
---

# Football Silver Schema Reference

## 1. Medallion Architecture

```
Bronze → Silver → Gold
```

| Layer | What lives here | Key rule |
|-------|----------------|----------|
| **Bronze** | Raw JSONP files exactly as received; JSONB dumps | Never modified after ingestion |
| **Silver** | All 10 relational tables below | Source of truth for all queries |
| **Gold** | `team_season_stats`, `player_season_stats`, `match_summaries`, `model_features` | Fully derived from silver; can be recomputed any time |

Gold is never a source of truth. Never bypass silver to query bronze.

---

## 2. Entity Tiers

```
Tier 1 — Competition framework
  competitions → competition_seasons ← seasons

Tier 2 — Participants
  teams → team_competition_seasons ← competition_seasons
  players → player_squads ← teams, competition_seasons

Tier 3 — Match data
  competition_seasons → matches ← teams (home + away)
  matches → match_lineups ← players, teams
  matches → events ← players, teams
```

**`competition_season_id` is the central FK** that anchors all season-specific entities.

---

## 3. Table Reference

### 3.1 `competitions`
Timeless club/competition identity. Never changes. Inserted once.

| Column | Type | Notes |
|--------|------|-------|
| competition_id | INT PK | Surrogate |
| source_competition_id | VARCHAR(50) UNIQUE | Provider ID — deduplication key |
| name | VARCHAR(100) | e.g. `Primera División` |
| known_name | VARCHAR(100) | e.g. `Spanish La Liga` |
| competition_code | VARCHAR(10) | e.g. `PRD` |
| competition_format | VARCHAR(50) | e.g. `Domestic league` |
| country | VARCHAR(60) | |
| confederation | VARCHAR(10) | `UEFA`, `CONMEBOL`, etc. |
| tier_level | INT | 1=top flight, 2=second div |

---

### 3.2 `seasons`
Label table only. One row per season label, shared across all competitions.

| Column | Type | Notes |
|--------|------|-------|
| season_id | INT PK | Surrogate |
| label | VARCHAR(10) UNIQUE | e.g. `2025/2026` — ingestion lookup key |

---

### 3.3 `competition_seasons`
**Central anchor of the model.** One row = one specific edition of a competition.

| Column | Type | Notes |
|--------|------|-------|
| competition_season_id | INT PK | Master FK referenced by all season-specific entities |
| source_season_id | VARCHAR(50) UNIQUE | Provider `tournamentCalendar.id` |
| source_stage_id | VARCHAR(50) UNIQUE | Provider `stage.id` — **primary ingestion lookup key** |
| competition_id | INT FK | → competitions |
| season_id | INT FK | → seasons |
| status | VARCHAR(20) | `upcoming` / `active` / `completed` |
| start_date, end_date | DATE | Competition calendar dates |
| stage_start_date, stage_end_date | DATE | Stage-level dates |
| num_teams, total_matchdays | INT | Manual |
| promo_spots, relegation_spots | INT | Manual |

> **Ingestion**: match files carry `stage.id`. Resolve via:
> `WHERE cs.source_stage_id = :source_stage_id`

---

### 3.4 `teams`
Stable club identity. Inserted once, never modified (provider-sourced fields only).

| Column | Type | Notes |
|--------|------|-------|
| team_id | INT PK | Surrogate |
| source_team_id | VARCHAR(50) UNIQUE | Provider `contestant.id` — deduplication key |
| source_venue_id | VARCHAR(50) | For future venue enrichment |
| name | VARCHAR(100) | Full official name |
| short_name | VARCHAR(30) | e.g. `Barça` |
| abbreviation | VARCHAR(5) | e.g. `FCB` |
| city, stadium_name | VARCHAR | Manual enrichment |
| stadium_capacity, founded_year | INT | Manual enrichment |
| country | VARCHAR(60) | |

---

### 3.5 `team_competition_seasons`
Season enrollment. New row each season — promotions/relegations handled naturally.

| Column | Type | Notes |
|--------|------|-------|
| team_cs_id | INT PK | |
| team_id | INT FK | → teams |
| competition_season_id | INT FK | → competition_seasons |
| kit_home_color, kit_away_color | VARCHAR(20) | Hex, manual |
| badge_url | VARCHAR(255) | Season-specific |

> **Unique constraint**: `(team_id, competition_season_id)`

---

### 3.6 `players`
Stable player identity. `full_name` is a **generated stored column** — never include in INSERT.

| Column | Type | Notes |
|--------|------|-------|
| player_id | SERIAL PK | |
| source_player_id | VARCHAR(50) UNIQUE | Deduplication key |
| first_name, last_name | VARCHAR | Provider |
| short_first_name, short_last_name | VARCHAR | Provider |
| full_name | VARCHAR GENERATED | `first_name || ' ' || last_name` — never write this |
| known_name | VARCHAR(100) | Public alias e.g. `Vinicius Jr.` |
| match_name | VARCHAR(150) | Used in event feed |
| gender | VARCHAR(10) | |
| nationality, nationality_source_id | VARCHAR | |
| second_nationality, second_nationality_source_id | VARCHAR | |
| position_raw | VARCHAR(30) | Provider broad: `Goalkeeper|Defender|Midfielder|Attacker` |
| preferred_position | VARCHAR(10) | Manual: `GK|CB|LB|RB|DM|CM|AM|LW|RW|ST` |
| preferred_foot | VARCHAR(5) | Manual: `left|right|both` |
| height_cm | INT | Manual |
| date_of_birth | DATE | Manual (not in provider) |

> Coaching staff (`type != "player"`) are **never** inserted here.

---

### 3.7 `player_squads`
Player's membership of a team per season. Mid-season transfers = two rows (first spell closed, new one opened).

| Column | Type | Notes |
|--------|------|-------|
| squad_id | SERIAL PK | |
| player_id | INT FK | → players |
| team_id | INT FK | → teams |
| competition_season_id | INT FK | → competition_seasons |
| start_date | DATE | Initial: provider `startDate`. Diff: snapshot date |
| end_date | DATE | Null if still active |
| shirt_number | INT | |
| squad_role | VARCHAR(20) | Default `first_team`; manual: `loan|youth` |
| transfer_type | VARCHAR(20) | Manual: `permanent|loan|free|youth` |

**Temporal join pattern** — find team on a given match date:
```sql
JOIN player_squads ps ON ps.player_id = e.player_id
WHERE match_date BETWEEN ps.start_date AND COALESCE(ps.end_date, '9999-12-31')
```

**Diff strategy** across snapshot loads:
1. Get `active_ids` = `source_player_id` WHERE `end_date IS NULL` for `(team, season)`
2. `departed = active_ids - incoming_ids` → UPDATE `end_date = snapshot_date`
3. `new_arrivals = incoming_ids - active_ids` → INSERT new row with `start_date = snapshot_date`
4. `continuing = incoming_ids ∩ active_ids` → UPDATE `shirt_number` in place if changed

---

### 3.8 `matches`
One row per match. Home/away stored as direct columns.

| Column | Type | Notes |
|--------|------|-------|
| match_id | SERIAL PK | |
| source_match_id | VARCHAR(50) UNIQUE | Provider `matchInfo.id` — deduplication key |
| competition_season_id | INT FK | Resolved via `source_stage_id` |
| home_team_id | INT FK | → teams |
| away_team_id | INT FK | → teams |
| matchday | INT | `matchInfo.week` |
| match_date | TIMESTAMP | Local Spain time (no timezone) |
| status | VARCHAR(20) | Always `completed` at load |
| winner | VARCHAR(10) | `home|away|draw` |
| home_score, away_score | INT | Full time |
| home_score_ht, away_score_ht | INT | Half time |
| match_length_min, match_length_sec | INT | Actual duration including stoppages |
| ht_injury_time_sec, ft_injury_time_sec | INT | Announced added time |
| var_used | BOOL | `matchInfo.var == "1"` |
| venue | VARCHAR(100) | |
| venue_latitude, venue_longitude | FLOAT | |
| neutral_venue | BOOL | |
| coverage_level | VARCHAR(10) | Data quality signal |

**Useful WHERE patterns**:
```sql
-- All matches for a team
WHERE home_team_id = X OR away_team_id = X

-- All matches in a season
WHERE competition_season_id = X

-- Winner lookup (more ergonomic than comparing scores)
WHERE winner = 'home'
```

---

### 3.9 `match_lineups`
One row per player per match. Separated from events because it's a squad fact, not an event.

| Column | Type | Notes |
|--------|------|-------|
| lineup_id | SERIAL PK | |
| match_id | INT FK | → matches |
| player_id | INT FK | → players |
| team_id | INT FK | → teams |
| starting_xi | BOOL | True if formation_slot > 0 from Q131 |
| formation_position | SMALLINT | 1-11 for starters; Q145 for subs; NULL unused bench |
| position_code | SMALLINT | Raw Q44 code: 1=GK 2=DEF 3=MID 4=FWD 5=SUB |
| position | VARCHAR(5) | Mapped: `GK|DEF|MID|FWD|SUB` |
| shirt_number | SMALLINT | From Q59 |
| is_captain | BOOL | source_player_id == Q194 value |
| team_formation | VARCHAR(15) | Raw Q130 formation ID (denormalised, repeated per team) |
| minute_in | SMALLINT | 0=starter; timeMin of typeId:19; NULL=unused sub |
| minute_out | SMALLINT | timeMin of exit event; NULL=played to end or never entered |
| exit_reason | VARCHAR(15) | `substitution|red_card|second_yellow|NULL` |
| raw_status_flag | SMALLINT | Q227 value (meaning unknown) |
| raw_sub_flag | SMALLINT | Q293 value from typeId:19 (meaning unknown) |

> **Unique constraint**: `(match_id, player_id)`
> **Conflict strategy**: `ON CONFLICT DO UPDATE` (providers issue lineup corrections post-match)

**5-Pass ingestion strategy** (load_match_lineups.py):

| Pass | Event typeId | Action |
|------|-------------|--------|
| 1 | 34 (Team set up) | INSERT all rows for both teams. Starters: `minute_in=0`. Bench: `minute_in=NULL`. |
| 2a | 18 (Player off) | UPDATE `minute_out=timeMin`, `exit_reason='substitution'` |
| 2b | 19 (Player on) | UPDATE `minute_in=timeMin`, `formation_position=Q145` |
| 2c | 17 (Card) + Q33 | UPDATE `minute_out=timeMin`, `exit_reason='red_card'` |
| 2d | 17 (Card) + Q32 | UPDATE `minute_out=timeMin`, `exit_reason='second_yellow'` |

Pass 2a **must** run before Pass 2b. Events are sorted by `eventId` before processing.

**Minutes played formula** (no re-scan of events needed):
```sql
COALESCE(ml.minute_out, m.match_length_min) - COALESCE(ml.minute_in, 0) AS minutes_played
FROM match_lineups ml
JOIN matches m USING (match_id)
```

---

### 3.10 `events`
Granular event stream. One row per event including synthesised carries.

Key columns (see `opta-events-reference` skill for full detail):
- `event_type` — human label (`Pass`, `Goal`, `Carry`, etc.)
- `type_id` — raw numeric (-1 for carries)
- `json_index` - Event order inside each match
- `x, y` — real pitch metres (already scaled from provider 0-100)
- `end_x, end_y` — destination
- `xt` — Expected Threat delta (Pass and Carry only)
- `sequence_id` — possession chain ID (populated in second pass)
- `spadl_type_id`  - SPADL mapping for action type
- `spadl_result_id`  - SPADL mapping for result type
- `spadl_bodypart_id` - SPADL mapping for bodypart
- `xg` - xg value of the shots 
- `vaep_offensive` - VAEP offensive value of the action
- `vaep_deffensive` - VAEP deffensive value of the action
- `vaep_value`- VAEP total value of the action
- `raw_data` — full provider JSONB (NULL for carries)

---

**IMPORTANT**   json_index = position in the raw JSONP array, which is the true chronological order. Carries get fractional json_index values so they sort between their surrounding events.

## 4. FK Resolution Pattern

**Never resolve source IDs in application code.** Always use SQL JOIN:

```sql
-- Example: insert a match, resolving all three FKs in one SELECT
INSERT INTO silver.matches (competition_season_id, home_team_id, away_team_id, ...)
SELECT
    cs.competition_season_id,
    ht.team_id,
    at.team_id,
    ...
FROM
    silver.competition_seasons cs,
    silver.teams ht,
    silver.teams at
WHERE
    cs.source_stage_id    = :source_stage_id
    AND ht.source_team_id = :home_source_team_id
    AND at.source_team_id = :away_source_team_id
ON CONFLICT (source_match_id) DO NOTHING;
```

This pattern is used consistently in `load_matches.py`, `load_teams.py`, `db.py` (squads).

---

## 5. Idempotency Patterns

| Table | Conflict strategy | Notes |
|-------|-----------------|-------|
| competitions | `DO NOTHING` on source_competition_id | Inserted once |
| seasons | `DO NOTHING` on label | Inserted once |
| competition_seasons | `DO NOTHING` on source_stage_id | Inserted once |
| teams | `DO NOTHING` on source_team_id | Inserted once |
| team_competition_seasons | `DO NOTHING` on (team_id, competition_season_id) | |
| players | `DO UPDATE` provider-sourced fields | Safely re-runnable; never overwrites manual fields |
| player_squads | Insert new rows only; close via UPDATE | Managed by diff logic |
| matches | `DO NOTHING` on source_match_id | |
| match_lineups | `DO UPDATE` on (match_id, player_id) | Providers issue corrections |
| events | No conflict key (insert all) | Guarded by `is_match_already_loaded` check |

---

## 6. squad_snapshot_log (Audit Table)

Tracks which squad files have been processed. Used as idempotency guard.

```sql
UNIQUE (source_team_id, source_season_id, snapshot_date)
```

Columns: `log_id, source_team_id, source_season_id, competition_code, season, snapshot_date, team_name, players_upserted, entries_inserted, entries_closed, processed_at`

---

## 7. Scalability Without Schema Changes

| Change | Operation |
|--------|-----------|
| New league | INSERT into competitions; reuse existing seasons rows |
| New season | INSERT into seasons + competition_seasons; new team_competition_seasons for promotions |
| Mid-season transfer | Close old player_squads row (set end_date); INSERT new row |
| New event types | Add to event_type values; raw payload preserved in raw_data |
| Cup competitions | New competition_id + competition_season rows; matches follow same schema |

---

## 8. Load Order Requirements

These dependencies must be satisfied before loading:

```
competitions, seasons
    → competition_seasons
        → team_competition_seasons (requires teams)
        → matches (requires teams)
            → match_lineups (requires players)
            → events (requires players, teams)
players, teams (independent)
```

`matches` must exist before `match_lineups` or `events` for that match can be loaded. Files for unresolved matches are **skipped and logged**, not treated as errors.