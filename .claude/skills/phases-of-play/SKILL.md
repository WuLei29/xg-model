---
name: phases-of-play
description: >
 Complete guide to understand the phases of play definitions and conditions used in the football analytics project
---

# Sequence Phases of Play — Design Specification

**Context:** `gold.sequences` table enrichment — phase-of-play classification  
**Status:** Design finalised, ready for implementation

---

## 1. Core Concept

A possession sequence can traverse **multiple phases of play** as it progresses across the pitch. Phases are not mutually exclusive — a single sequence can be tagged with several phases (e.g. *build-up → mid-block progression → attacking 3rd*). The classification operates on two independent axes:

- **Zone** — which third of the pitch the action occurs in (defensive, middle, attacking).
- **Tempo** — whether the possession is *established* (≥ 3 passes/carries in that zone) or *fast* (< 3, moved through quickly).

Additionally, origin-based and structural phases (counter-attack, set piece, direct long play) overlay the zone×tempo grid.

---

## 2. Phase Definitions

### 2.1 Zone × Tempo Phases

These six phases form the core grid. Every segment of a sequence that involves at least one event in a zone gets classified on both axes.

| Phase | Zone | Tempo | Definition |
|---|---|---|---|
| **Build-up** | Zone 1 (0–35m) | Established (≥ 3 passes/carries in zone) | Patient possession in the defensive third. Can be *static* (from goal-kick, free-kick) or *dynamic* (from open-play recovery). |
| **Fast Build-up** | Zone 1 (0–35m) | Fast (< 3 passes/carries before leaving zone) | Rapid progression out of the defensive third. |
| **Mid-block Progression** | Zone 2 (35–70m) | Established (≥ 3 passes/carries in zone) | Sustained possession through the middle third. |
| **Fast Mid-block** | Zone 2 (35–70m) | Fast (< 3 passes/carries before leaving zone, going to zone 3) | Quick transition through the middle third without settling. |
| **Attacking 3rd** | Zone 3 (70–105m) | Established (≥ 3 passes/carries in zone) | Sustained possession in the final third — combination play, crossing sequences, build-up to a shot. |
| **Fast Attacking 3rd** | Zone 3 (70–105m) | Fast (< 3 passes/carries in zone before ending the sequence) | Rapid entry into the final third ending in a shot, cross, or loss of possession before establishing control. |

**Tempo threshold:** The ≥ 3 passes/carries threshold is the working definition of "established possession" in a zone. This is a tunable parameter.

### 2.2 Origin-Based Phases

These phases are defined by *how* the sequence started, combined with behavioural thresholds.

| Phase | Origin | Conditions |
|---|---|---|
| **Counter-attack** | Ball won back in **own half** (zone 1 or zone 2, x < 52.5m) via tackle, interception, ball recovery, or turnover. | ≥ 75% directionality towards goal. Travelled ≥ 18 yards (~16.5m) towards goal. Duration ≤ 15s or ≤ 5 passes. |
| **High Transition** | Ball won back in **opponent's half** (zone 2 or zone 3, x ≥ 52.5m) via tackle, interception, ball recovery, or turnover. | Quick forward play after recovery. Same directionality and tempo logic as counter-attack but from a high-press recovery. |

**Key distinction from "fast break":** Fast break describes *tempo through a zone* (an attribute of any sequence). Counter-attack and high transition describe *how possession was won* — they are origin classifications. A sequence can be both a counter-attack *and* contain fast mid-block + fast attacking 3rd phases.

**Why no generic "transition" flag:** Whether a sequence originates from a turnover is already derivable from `start_type` (Tackle, Interception, BallRecovery) + `start_zone`. A dedicated boolean flag would be redundant. Counter-attack and high transition add value because they combine origin with behavioural thresholds (directionality, speed, distance) that aren't derivable from existing columns.

### 2.3 Structural Phases

| Phase | Definition |
|---|---|
| **Set Piece** | Sequence involving a corner, free-kick, or long throw-in directed into the penalty area. Phase ends when: (a) opposition establishes clear possession, (b) ball cleared to defending team's half, (c) ball goes out of play, or (d) 20 seconds have elapsed since the set piece was taken. |
| **Direct Long Play** | A single event within a sequence where the ball is played ≥ 32m forward on the x-axis with an angle ≤ 30° from the goal direction (~75% direct towards goal). Can occur within any other phase. |
| **Direct FK / Penalty** | A goal scored directly from a free-kick or penalty. These are trivially classified single-event sequences. |

**Set piece exclusions:** Quick free-kicks in the team's own half, free-kicks without a cross attempt, and free-kicks with the ball inside the penalty area taken within 10 seconds are **not** classified as set pieces. These are reclassified as the start of build-up, mid-block progression, or attacking 3rd depending on the zone.

**"After set piece" phase — dropped:** If a set piece phase ends but possession continues, the sequence transitions into whatever the next zone×tempo phase is. The phase segment table captures this transition naturally — no dedicated "after set piece" category is needed.

### 2.4 Chaotic

Sequences that do not match any of the previous criteria stated before.

---

## 3. Edge Cases

### 3.1 Zone Re-entry (Ball Returned to Previous Zone)

When the ball briefly enters a new zone but returns to the previous zone (e.g. a pass into zone 2 that gets played back to zone 1), the phase should **not** change. The current phase is maintained.

**Rule:** A zone transition is only confirmed when the ball *stays* in the new zone for at least 3  events, so it can be classified in stablished possesion or fast-break. If not, it will be considered of part of the current phase. Few events in a new zone followed by an immediate return to the previous zone does not trigger a phase change — it is absorbed into the current phase.

**Implementation hint:** The state machine should use a "pending zone change" buffer. When the ball enters a new zone, mark it as pending. If the next event is still in the new zone, confirm the transition. If it returns to the previous zone, discard the pending change and continue the current phase.

### 3.2 Single-Event Sequences

Sequences with only 1 event (e.g. a clearance, a long ball immediately lost) may not have enough events to classify a meaningful phase. These should be tagged with the zone of the single event and marked as "other" by default.

### 3.3 Set Piece into Open Play

A set piece (corner, free-kick) that leads to sustained possession in the attacking third is two phases: *set piece → attacking 3rd*. The 20-second window or the exclusion criteria determine where the set piece phase ends and the open-play phase begins.

### 3.4 Counter-Attack That Slows Down

A sequence that starts as a counter-attack (turnover + fast progression) but then settles into established possession in a zone loses the "counter-attack" tempo character but retains the origin flag. The boolean `has_counter_attack` on the sequence is still `true`, but the phase segments show the transition from counter-attack speed to established play.

---

## 4. Data Strategy

### 4.1 Table A: Phase Flags on `gold.sequences`

Boolean columns added directly to the `gold.sequences` table. One column per phase. A sequence gets `true` if it *contains* that phase at any point.

```
sequence_id | has_buildup | has_fast_buildup | has_midblock | has_fast_midblock | has_attacking | has_fast_attacking | has_set_piece | has_counter_attack | has_high_transition | has_direct_long | has_direct_fk_pk
```

**Use case:** Fast filtering and aggregation. "What % of sequences involve a counter-attack?" "Show me all sequences with established attacking 3rd play."

### 4.2 Table B: `gold.sequence_phase_segments`

One row per contiguous phase within a sequence. Captures the internal structure and enables transition analysis.

| Column | Type | Description |
|---|---|---|
| `sequence_id` | VARCHAR (FK) | Reference to `gold.sequences` |
| `phase_order` | INT | Ordinal position of this phase within the sequence (1, 2, 3…) |
| `phase_type` | VARCHAR | Phase name: `buildup`, `fast_buildup`, `midblock`, `fast_midblock`, `attacking`, `fast_attacking`, `set_piece`, `counter_attack`, `high_transition`, `direct_long`, `direct_fk_pk` |
| `start_event_id` | BIGINT (FK) | First event in this phase segment |
| `end_event_id` | BIGINT (FK) | Last event in this phase segment |
| `event_count` | INT | Number of events in this phase segment |
| `start_x` | FLOAT | x-coordinate of the first event in this segment |
| `end_x` | FLOAT | x-coordinate of the last event in this segment |
| `duration_seconds` | FLOAT | Duration of this phase segment |

**Use case:** Transition analysis. Build a transition matrix by querying consecutive `phase_order` values:

```sql
SELECT
    curr.phase_type  AS from_phase,
    next.phase_type  AS to_phase,
    COUNT(*)         AS transition_count
FROM gold.sequence_phase_segments curr
JOIN gold.sequence_phase_segments next
  ON curr.sequence_id = next.sequence_id
 AND next.phase_order = curr.phase_order + 1
GROUP BY 1, 2
ORDER BY 3 DESC;
```

### 4.3 Implementation Order

1. **Phase flags on `gold.sequences`** — cheap to compute, immediately useful for filtering and clustering. Requires walking through events and counting passes/carries per zone.
2. **Phase segmentation (`gold.sequence_phase_segments`)** — requires a state machine that tracks zone, tempo, and zone re-entry edge cases. More complex but enables the transition analysis.
3. **Transition matrix** — pure SQL on top of the segments table. No additional computation needed.

---

## 5. Phase Detection Logic — State Machine Outline

The phase classifier operates **within** a single sequence (unlike the sequence classifier which operates across the full event stream). It receives the possessing team's events for one sequence, sorted chronologically.

### 5.1 State Variables

```
current_zone:         INT (1, 2, 3)       — derived from event x-coordinate
current_phase_start:  event_id            — first event of the current phase
events_in_zone:       INT                 — passes/carries counted in current zone
pending_zone_change:  Optional[INT]       — buffered zone if ball just entered a new zone
pending_event:        Optional[event_id]  — the event that triggered the pending change
```

### 5.2 Per-Event Logic (Pseudocode)

```
for each event in sequence:
    event_zone = classify_zone(event.x)

    if event_zone != current_zone:
        if pending_zone_change == event_zone:
            # Confirmed: ball stayed in new zone for 2+ events
            emit_phase_segment(current_phase_start, previous_event)
            current_zone = event_zone
            current_phase_start = pending_event
            events_in_zone = 2  # the pending event + this one
            pending_zone_change = None
        elif pending_zone_change is not None and event_zone == current_zone:
            # Ball returned to previous zone — discard pending change
            pending_zone_change = None
            events_in_zone += 1  # count normally
        else:
            # First event in a new zone — mark as pending
            pending_zone_change = event_zone
            pending_event = event
    else:
        if pending_zone_change is not None and event_zone == current_zone:
            # Ball returned to current zone — discard pending
            pending_zone_change = None
        events_in_zone += 1

    # Count only passes and carries for tempo classification
    if event.type in ('Pass', 'Carry'):
        passes_carries_in_zone += 1
```

### 5.3 Tempo Classification

When a phase segment is emitted (because the zone changed or the sequence ended), classify its tempo:

- `passes_carries_in_zone >= 3` → established (build-up / mid-block / attacking 3rd)
- `passes_carries_in_zone < 3` & progression to the next zone → fast (fast build-up / fast mid-block / fast attacking 3rd)

### 5.4 Origin-Based Overlay

After the zone×tempo segments are computed, check if the sequence qualifies for counter-attack or high transition:

1. Check `start_type` — must be one of: Tackle, Interception, BallRecovery (turnover).
2. Check `start_zone` — own half (x < 52.5m) for counter-attack, opponent's half for high transition.
3. Compute directionality across the full sequence (≥ 75% towards goal).
4. Check distance (≥ 16.5m forward) and tempo (≤ 15s or ≤ 5 passes for counter-attack; ≤ 20s or ≤ 8 passes for high transition).

These are applied as additional boolean flags, not as segment types — a counter-attack sequence still has zone×tempo segments internally.

---

## 6. Complete Phase Catalogue

| # | Phase | Type | Zone | Trigger |
|---|---|---|---|---|
| 1 | Build-up | Zone × Tempo | Zone 1 | ≥ 3 passes/carries in defensive third |
| 2 | Fast Build-up | Zone × Tempo | Zone 1 | < 3 passes/carries, exited zone quickly |
| 3 | Mid-block Progression | Zone × Tempo | Zone 2 | ≥ 3 passes/carries in middle third |
| 4 | Fast Mid-block | Zone × Tempo | Zone 2 | < 3 passes/carries, exited zone quickly |
| 5 | Attacking 3rd | Zone × Tempo | Zone 3 | ≥ 3 passes/carries in attacking third |
| 6 | Fast Attacking 3rd | Zone × Tempo | Zone 3 | < 3 passes/carries in attacking third |
| 7 | Set Piece | Structural | Any (usually zone 3) | Corner, free-kick, or long throw-in into the box. 20s window. Exclusions apply. |
| 8 | Counter-attack | Origin | Starts own half | Turnover + ≥ 75% directionality + ≥ 16.5m forward + ≤ 15s or ≤ 5 passes |
| 9 | High Transition | Origin | Starts opp. half | Turnover in opponent's half + quick forward play |
| 10 | Direct Long Play | Structural | Any | Single event ≥ 32m forward, ≤ 30° from goal axis |
| 11 | Direct FK / Penalty | Structural | Zone 3 | Goal directly from free-kick or penalty |

---

## 7. Dependencies

### 7.1 Required from `gold.sequences`

The phase classifier needs these columns already computed:

- `sequence_id`, `match_id`, `team_id`
- `start_event_id`, `end_event_id`
- `start_type` (for counter-attack / high transition origin check)
- `start_zone` (for origin zone classification)

### 7.2 Required from `silver.events`

The phase classifier reads event-level data for each sequence:

- `event_id`, `event_type`, `outcome`
- `x`, `y`, `end_x`, `end_y` (scaled to 105×68)
- `team_id`
- `minute`, `second`, `timestamp` (for duration and tempo calculations)
- `raw_data` (for set piece qualifier extraction — corners, free-kicks, throw-ins)

### 7.3 Pre-materialisation

Before running the phase classifier, qualifier flags must be extracted from `raw_data` for set piece detection, similar to how `preprocess_for_sequences` materialises `is_set_piece_pass` and `is_dead_ball_goal`. Required flags:

- `is_corner` (qualifier ID 6)
- `is_free_kick` (qualifier ID 5)
- `is_throw_in` (qualifier ID 107)
- `is_cross` (for set piece exclusion logic — free-kicks without cross attempt)