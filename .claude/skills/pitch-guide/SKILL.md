---
name: pitch-guide
description: >
  Language-agnostic football pitch specification: all geometric primitives for
  drawing a 105x68m pitch (lines, arcs, circles, rectangles with exact coordinates
  and angles), the 30-zone grid (6 longitudinal strips x 5 lateral channels) with
  boundaries aligned to pitch markings, zone naming conventions, assignment logic
  in Python and SQL, and tactical context for each zone group.

  Use this skill whenever drawing or rendering a football pitch in any language or
  framework (Python/matplotlib, JS/Canvas, SVG, D3, React, etc.), implementing zone
  assignment in pipelines or gold-layer SQL, building heatmaps or spatial
  visualisations, or answering questions about pitch coordinates, zones, or channels.
---

# Pitch Guide — Drawing Specification & Zone Reference

## 1. Pitch Coordinate System

Standard **105 x 68 m** pitch. All coordinates in real-world metres.

### Axes

| Axis | Range | Direction |
|------|-------|-----------|
| **x** | 0 -> 105 | Left goal line -> Right goal line (length) |
| **y** | 0 -> 68 | Bottom touchline -> Top touchline (width) |

### Origin & Orientation

- **(0, 0)** = bottom-left corner (left goal line, bottom touchline)
- **(105, 68)** = top-right corner (right goal line, top touchline)
- Attacking direction: left to right (increasing x)

### Coordinate Conversion from Opta

Opta raw data uses a 0-100 scale. Silver layer converts during ingestion:

```
x_metres = (opta_x / 100) * 105
y_metres = (opta_y / 100) * 68
```

---

## 2. Pitch Markings — Key Dimensions

All values derived from FIFA Laws of the Game for a 105x68m pitch.

| Landmark | Value |
|----------|-------|
| Pitch length | 105 m |
| Pitch width | 68 m |
| Centre spot | (52.5, 34.0) |
| Centre circle radius | 9.15 m |
| Halfway line | x = 52.5 |
| Goal width | 7.32 m |
| Goal depth | 2.0 m |
| Goal post y-positions | 30.34 and 37.66 |
| Penalty area depth | 16.5 m |
| Penalty area width | 40.32 m |
| Penalty area y-range | 13.84 -> 54.16 |
| Six-yard box depth | 5.5 m |
| Six-yard box width | 18.32 m |
| Six-yard box y-range | 24.84 -> 43.16 |
| Penalty spot distance | 11.0 m from goal line |
| Corner arc radius | 1.0 m |
| Penalty arc radius | 9.15 m (same as centre circle) |

### How Dimensions Relate

```
Goal width = 7.32 m, centred at y = 34.0
  -> goal posts at y = 30.34 and y = 37.66

Penalty area = 16.5 m either side of goal posts
  -> y: 30.34 - 16.5 = 13.84  to  37.66 + 16.5 = 54.16
  -> total width: 40.32 m

Six-yard box = 5.5 m either side of goal posts
  -> y: 30.34 - 5.5 = 24.84  to  37.66 + 5.5 = 43.16
  -> total width: 18.32 m
```

---

## 3. Pitch Drawing Primitives

Every element needed to render the pitch, expressed as geometric primitives. All coordinates in metres. Angles in degrees, measured counter-clockwise from the positive x-axis (standard mathematical convention).

### 3.1 Rendering Defaults

| Property | Value |
|----------|-------|
| Pitch fill | green (e.g. `#2d8a4e` or `#538032`) |
| Line colour | white (`#ffffff`) |
| Line width | 1-2 px (adjust for display scale) |
| Goal colour | white or light grey |
| Spot radius | 0.3 m (visual; actual is ~0.12 m) |

### 3.2 Boundary Lines

| Element | Type | Coordinates |
|---------|------|-------------|
| **Pitch outline** | Rectangle | (0, 0) to (105, 68) |
| **Halfway line** | Line | (52.5, 0) to (52.5, 68) |

### 3.3 Penalty Areas

| Element | Type | Coordinates |
|---------|------|-------------|
| **Left penalty area** | Rectangle | (0, 13.84) to (16.5, 54.16) |
| **Right penalty area** | Rectangle | (88.5, 13.84) to (105, 54.16) |

### 3.4 Six-Yard Boxes

| Element | Type | Coordinates |
|---------|------|-------------|
| **Left six-yard box** | Rectangle | (0, 24.84) to (5.5, 43.16) |
| **Right six-yard box** | Rectangle | (99.5, 24.84) to (105, 43.16) |

### 3.5 Goals

| Element | Type | Coordinates | Notes |
|---------|------|-------------|-------|
| **Left goal** | Rectangle | (-2, 30.34) to (0, 37.66) | Behind the goal line; optional |
| **Right goal** | Rectangle | (105, 30.34) to (107, 37.66) | Behind the goal line; optional |

### 3.6 Centre Mark & Circle

| Element | Type | Parameters |
|---------|------|------------|
| **Centre spot** | Circle (filled) | Centre: (52.5, 34.0), radius: 0.3 |
| **Centre circle** | Circle (stroke) | Centre: (52.5, 34.0), radius: 9.15 |

### 3.7 Penalty Spots

| Element | Type | Parameters |
|---------|------|------------|
| **Left penalty spot** | Circle (filled) | Centre: (11.0, 34.0), radius: 0.3 |
| **Right penalty spot** | Circle (filled) | Centre: (94.0, 34.0), radius: 0.3 |

### 3.8 Penalty Arcs

The penalty arc is the portion of a circle (r = 9.15 m, centred on the penalty spot) that falls **outside** the penalty area.

**Left penalty arc:**
- Centre: (11.0, 34.0), radius: 9.15
- Intersection with x = 16.5: cos(theta) = 5.5 / 9.15 = 0.6011 -> theta = 53.05 deg
- **Arc: from -53.05 deg to +53.05 deg** (the right-facing bulge outside the penalty area)

**Right penalty arc:**
- Centre: (94.0, 34.0), radius: 9.15
- Intersection with x = 88.5: cos(theta) = -5.5 / 9.15 = -0.6011 -> theta = 126.95 deg
- **Arc: from +126.95 deg to +233.05 deg** (the left-facing bulge outside the penalty area)

| Element | Centre | Radius | Start Angle | End Angle |
|---------|--------|--------|-------------|-----------|
| **Left penalty arc** | (11.0, 34.0) | 9.15 | -53.05 deg | +53.05 deg |
| **Right penalty arc** | (94.0, 34.0) | 9.15 | +126.95 deg | +233.05 deg |

### 3.9 Corner Arcs

Quarter circles at each corner, radius = 1.0 m.

| Corner | Centre | Radius | Start Angle | End Angle |
|--------|--------|--------|-------------|-----------|
| **Bottom-left** | (0, 0) | 1.0 | 0 deg | 90 deg |
| **Bottom-right** | (105, 0) | 1.0 | 90 deg | 180 deg |
| **Top-right** | (105, 68) | 1.0 | 180 deg | 270 deg |
| **Top-left** | (0, 68) | 1.0 | 270 deg | 360 deg |

### 3.10 Drawing Order (back to front)

1. Pitch fill (green rectangle)
2. Pitch outline
3. Halfway line
4. Centre circle
5. Penalty areas (left, right)
6. Six-yard boxes (left, right)
7. Penalty arcs (left, right)
8. Corner arcs (4x)
9. Centre spot
10. Penalty spots (left, right)
11. Goals (optional, behind goal lines)

---

## 4. Framework Translation Notes

### Canvas / JS (HTML5 Canvas)

```
arc(centreX, centreY, radius, startAngle, endAngle, anticlockwise)
```

Canvas `arc()` uses **radians** and measures clockwise from the positive x-axis (y-axis points down). If your canvas has y-axis flipped (0 at bottom), negate the angles or flip with a transform.

### SVG

SVG arcs use the endpoint parameterisation (`A rx ry x-rotation large-arc-flag sweep-flag x y`). Convert from centre+angles:

```
startX = cx + r * cos(startAngle)
startY = cy + r * sin(startAngle)
endX   = cx + r * cos(endAngle)
endY   = cy + r * sin(endAngle)
```

### matplotlib / mplsoccer

The project already uses `mplsoccer` with `pitch_type='custom', pitch_length=105, pitch_width=68`. When drawing custom overlays on top of the mplsoccer pitch, use these same coordinates directly.

### Coordinate Transform

If your rendering system uses a different origin or scale:

```
render_x = (pitch_x / 105) * canvas_width  + offset_x
render_y = (pitch_y / 68)  * canvas_height + offset_y
```

For y-axis-down systems (Canvas, SVG default), flip y:

```
render_y = canvas_height - (pitch_y / 68) * canvas_height + offset_y
```

---

## 5. Zone Division System

The pitch is divided into **30 zones**: 6 longitudinal strips (x-axis) x 5 lateral channels (y-axis). All boundaries align with actual pitch markings.

### 5.1 Longitudinal Strips (x-axis) — 6 Strips

The pitch is split into three thirds (each 35 m), then each third is subdivided into two strips aligned with pitch markings.

| Strip | x range | Width (m) | Name | Tactical meaning |
|-------|---------|-----------|------|------------------|
| **X1** | 0.00 -> 16.50 | 16.50 | Defensive box | Own penalty area depth |
| **X2** | 16.50 -> 35.00 | 18.50 | Defensive build-up | Rest of defensive third |
| **X3** | 35.00 -> 52.50 | 17.50 | Middle defensive | First half of middle third |
| **X4** | 52.50 -> 70.00 | 17.50 | Middle attacking | Second half of middle third |
| **X5** | 70.00 -> 88.50 | 18.50 | Attacking approach | Attacking third before the box |
| **X6** | 88.50 -> 105.00 | 16.50 | Attacking box | Opponent penalty area depth |

**Design rationale:** The subdivision boundaries at 16.5 m and 88.5 m align exactly with the penalty area lines, the most tactically significant longitudinal markings on the pitch. The halfway line (52.5 m) splits the middle third symmetrically.

```
|--- Defensive Third ---|--- Middle Third ---|--- Attacking Third ---|
|   X1    |     X2      |   X3    |    X4    |     X5      |   X6    |
0       16.5          35.0      52.5       70.0          88.5      105
          ^                      ^                        ^
     Penalty area           Halfway line            Penalty area
```

### 5.2 Lateral Channels (y-axis) — 5 Channels

The channels are defined by the penalty area edges (y = 13.84 and y = 54.16) and the six-yard box edges (y = 24.84 and y = 43.16). Two wide flanks, two half-spaces, one central corridor.

| Channel | y range | Width (m) | ID | Name |
|---------|---------|-----------|-----|------|
| **Y1** | 0.00 -> 13.84 | 13.84 | `wide_left` | Wide channel (left touchline) |
| **Y2** | 13.84 -> 24.84 | 11.00 | `half_space_left` | Left half-space |
| **Y3** | 24.84 -> 43.16 | 18.32 | `center` | Central corridor (six-yard box width) |
| **Y4** | 43.16 -> 54.16 | 11.00 | `half_space_right` | Right half-space |
| **Y5** | 54.16 -> 68.00 | 13.84 | `wide_right` | Wide channel (right touchline) |

**Design rationale:** The central corridor (Y3) spans the full width of the six-yard box (18.32 m), capturing the most dangerous scoring zone. The half-spaces (Y2, Y4) correspond to the areas between the six-yard box and the penalty area edges. The wide channels (Y1, Y5) cover everything outside the penalty area width.

```
y=0                                                           y=68
|     Y1      |    Y2     |       Y3        |    Y4     |     Y5      |
|  wide_left  | hs_left   |     center      | hs_right  | wide_right  |
0           13.84       24.84             43.16       54.16          68
               ^           ^               ^           ^
          Pen. area    Six-yard box    Six-yard box   Pen. area
           edge          edge            edge          edge
```

---

## 6. Zone Naming Convention

Each zone is identified by a composite key: `X{strip}_{channel_name}`.

### Full Zone Grid (30 zones)

|  | Y1: wide_left | Y2: half_space_left | Y3: center | Y4: half_space_right | Y5: wide_right |
|--|---------------|---------------------|------------|----------------------|----------------|
| **X1** (0-16.5) | X1_wide_left | X1_half_space_left | X1_center | X1_half_space_right | X1_wide_right |
| **X2** (16.5-35) | X2_wide_left | X2_half_space_left | X2_center | X2_half_space_right | X2_wide_right |
| **X3** (35-52.5) | X3_wide_left | X3_half_space_left | X3_center | X3_half_space_right | X3_wide_right |
| **X4** (52.5-70) | X4_wide_left | X4_half_space_left | X4_center | X4_half_space_right | X4_wide_right |
| **X5** (70-88.5) | X5_wide_left | X5_half_space_left | X5_center | X5_half_space_right | X5_wide_right |
| **X6** (88.5-105) | X6_wide_left | X6_half_space_left | X6_center | X6_half_space_right | X6_wide_right |

### Compact Numeric Notation

For database columns, CSV exports, or ML feature vectors, use `zone_XY` where X = strip (1-6) and Y = channel (1-5):

```
zone_11  zone_12  zone_13  zone_14  zone_15
zone_21  zone_22  zone_23  zone_24  zone_25
zone_31  zone_32  zone_33  zone_34  zone_35
zone_41  zone_42  zone_43  zone_44  zone_45
zone_51  zone_52  zone_53  zone_54  zone_55
zone_61  zone_62  zone_63  zone_64  zone_65
```

---

## 7. Zone Assignment Logic

### Python Reference Implementation

```python
X_BOUNDARIES = [0, 16.5, 35.0, 52.5, 70.0, 88.5, 105.0]
Y_BOUNDARIES = [0, 13.84, 24.84, 43.16, 54.16, 68.0]

CHANNEL_NAMES = ['wide_left', 'half_space_left', 'center', 'half_space_right', 'wide_right']
STRIP_NAMES  = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']


def assign_zone(x: float, y: float) -> str:
    x_strip = None
    for i in range(len(X_BOUNDARIES) - 1):
        if X_BOUNDARIES[i] <= x <= X_BOUNDARIES[i + 1]:
            x_strip = i + 1
            break

    y_channel = None
    for j in range(len(Y_BOUNDARIES) - 1):
        if Y_BOUNDARIES[j] <= y <= Y_BOUNDARIES[j + 1]:
            y_channel = j + 1
            break

    if x_strip is None or y_channel is None:
        return 'off_pitch'

    return f"{STRIP_NAMES[x_strip - 1]}_{CHANNEL_NAMES[y_channel - 1]}"
```

### SQL Reference (for gold-layer views)

```sql
CASE
    WHEN x BETWEEN 0    AND 16.5  THEN 1
    WHEN x BETWEEN 16.5 AND 35.0  THEN 2
    WHEN x BETWEEN 35.0 AND 52.5  THEN 3
    WHEN x BETWEEN 52.5 AND 70.0  THEN 4
    WHEN x BETWEEN 70.0 AND 88.5  THEN 5
    WHEN x BETWEEN 88.5 AND 105.0 THEN 6
END AS x_strip,

CASE
    WHEN y BETWEEN 0     AND 13.84 THEN 1  -- wide_left
    WHEN y BETWEEN 13.84 AND 24.84 THEN 2  -- half_space_left
    WHEN y BETWEEN 24.84 AND 43.16 THEN 3  -- center
    WHEN y BETWEEN 43.16 AND 54.16 THEN 4  -- half_space_right
    WHEN y BETWEEN 54.16 AND 68.0  THEN 5  -- wide_right
END AS y_channel
```

### JavaScript Reference

```javascript
const X_BOUNDARIES = [0, 16.5, 35.0, 52.5, 70.0, 88.5, 105.0];
const Y_BOUNDARIES = [0, 13.84, 24.84, 43.16, 54.16, 68.0];
const CHANNEL_NAMES = ['wide_left', 'half_space_left', 'center', 'half_space_right', 'wide_right'];
const STRIP_NAMES = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6'];

function assignZone(x, y) {
    let xStrip = null;
    for (let i = 0; i < X_BOUNDARIES.length - 1; i++) {
        if (x >= X_BOUNDARIES[i] && x <= X_BOUNDARIES[i + 1]) {
            xStrip = i;
            break;
        }
    }
    let yChannel = null;
    for (let j = 0; j < Y_BOUNDARIES.length - 1; j++) {
        if (y >= Y_BOUNDARIES[j] && y <= Y_BOUNDARIES[j + 1]) {
            yChannel = j;
            break;
        }
    }
    if (xStrip === null || yChannel === null) return 'off_pitch';
    return `${STRIP_NAMES[xStrip]}_${CHANNEL_NAMES[yChannel]}`;
}
```

### Edge Cases

- **Exact boundary values** (e.g. x = 16.5, y = 13.84): assigned to the **lower-index** zone by the `<=` on both sides. SQL `CASE` evaluates top-to-bottom and stops at the first match, so boundary values fall into the earlier strip/channel.
- **Coordinates outside 0-105 / 0-68**: return `'off_pitch'` in Python/JS; evaluate to `NULL` in SQL. Should not exist in clean silver data.

---

## 8. Tactical Context by Zone Group

### Longitudinal Groups

| Group | Strips | Usage in sequence analysis |
|-------|--------|---------------------------|
| **Own box** | X1 | Goal kicks, defensive clearances, GK distribution |
| **Build-up** | X2, X3 | Possession build-up, press resistance, progressive passing origin |
| **Transition** | X3, X4 | Central duels, midfield turnovers, counter-attack triggers |
| **Chance creation** | X5 | Final-third entries, crosses, through-ball origins |
| **Box entries** | X6 | Shots, key passes, penalty area actions |

### Lateral Groups

| Group | Channels | Usage in sequence analysis |
|-------|----------|---------------------------|
| **Wide** | Y1, Y5 | Wing play, crosses, overlapping runs |
| **Half-spaces** | Y2, Y4 | Positional play, diagonal runs, creative passing |
| **Central** | Y3 | Direct attacks, through balls, shots, defensive spine |

---

## 9. Zone Visualisation — Drawing Zones on the Pitch

To overlay the 30-zone grid on a pitch rendering, draw these dividing lines:

### Longitudinal Division Lines (vertical lines)

| Line | x | From | To |
|------|---|------|----|
| Penalty area (left) | 16.5 | (16.5, 0) | (16.5, 68) |
| Defensive/middle third | 35.0 | (35.0, 0) | (35.0, 68) |
| Halfway line | 52.5 | already drawn as pitch marking |
| Middle/attacking third | 70.0 | (70.0, 0) | (70.0, 68) |
| Penalty area (right) | 88.5 | (88.5, 0) | (88.5, 68) |

### Lateral Division Lines (horizontal lines)

| Line | y | From | To |
|------|---|------|----|
| Penalty area bottom | 13.84 | (0, 13.84) | (105, 13.84) |
| Six-yard box bottom | 24.84 | (0, 24.84) | (105, 24.84) |
| Six-yard box top | 43.16 | (0, 43.16) | (105, 43.16) |
| Penalty area top | 54.16 | (0, 54.16) | (105, 54.16) |

### Zone Label Positions (centre of each zone)

For placing text labels, use the midpoint of each zone's bounding box. Example for X3_center:

```
label_x = (35.0 + 52.5) / 2 = 43.75
label_y = (24.84 + 43.16) / 2 = 34.0
```

---

## 10. Usage in Gold-Layer Tables

### `gold.sequences` — Possession Sequences

Each sequence row should include:

- `start_zone` — zone of the first event in the sequence
- `end_zone` — zone of the last event in the sequence
- `start_x_strip`, `start_y_channel` — numeric components for easy filtering
- `end_x_strip`, `end_y_channel` — numeric components for easy filtering

### Zone Transition Matrix

For phase-of-play analysis, build a 30x30 transition matrix counting how often sequences move from one zone to another. This powers questions like: "How often do sequences starting in X2_half_space_left end in X6_center?"

### Per-Zone Aggregations

Zone-level stats (pass completion %, xT generated, defensive actions per zone) should use `x_strip` and `y_channel` as GROUP BY dimensions, allowing flexible roll-ups by third, channel, or individual zone.

---

## 11. Relationship to Previous Zone System

The old zone system used 20 zones with slightly different boundaries (y = 13.54, 24.54, 43.35, 54.35) that did not align with the actual pitch markings. The new system:

- **Aligns exactly** with penalty_area_bottom (13.84), six_yard_bottom (24.84), six_yard_top (43.16), and penalty_area_top (54.16).
- Increases granularity from 20 to 30 zones by using a consistent 6x5 grid.
- Introduces half-spaces as first-class channels, essential for modern positional-play analysis.
- Uses a regular naming convention (`X{strip}_{channel}`) that is sortable, groupable, and self-documenting.

---

## 12. Constants Summary

```python
# Copy-paste ready constants for any module

PITCH_LENGTH = 105
PITCH_WIDTH  = 68

# Zone boundaries
X_BOUNDARIES  = [0, 16.5, 35.0, 52.5, 70.0, 88.5, 105.0]
Y_BOUNDARIES  = [0, 13.84, 24.84, 43.16, 54.16, 68.0]
STRIP_NAMES   = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']
CHANNEL_NAMES = ['wide_left', 'half_space_left', 'center', 'half_space_right', 'wide_right']
N_STRIPS   = 6
N_CHANNELS = 5
N_ZONES    = 30

# Pitch marking coordinates
CENTRE_SPOT       = (52.5, 34.0)
CENTRE_CIRCLE_R   = 9.15
PENALTY_AREA_LEFT  = {'x': (0, 16.5),    'y': (13.84, 54.16)}
PENALTY_AREA_RIGHT = {'x': (88.5, 105),   'y': (13.84, 54.16)}
SIX_YARD_LEFT      = {'x': (0, 5.5),      'y': (24.84, 43.16)}
SIX_YARD_RIGHT     = {'x': (99.5, 105),    'y': (24.84, 43.16)}
PENALTY_SPOT_LEFT  = (11.0, 34.0)
PENALTY_SPOT_RIGHT = (94.0, 34.0)
GOAL_LEFT          = {'x': (-2, 0),        'y': (30.34, 37.66)}
GOAL_RIGHT         = {'x': (105, 107),     'y': (30.34, 37.66)}
GOAL_POST_Y        = (30.34, 37.66)
CORNER_ARC_R       = 1.0
PENALTY_ARC_R      = 9.15
PENALTY_ARC_ANGLE  = 53.05  # degrees from horizontal to intersection with penalty area line
```
