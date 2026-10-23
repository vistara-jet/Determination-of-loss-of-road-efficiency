# Capacity-loss methodology (IRC-based)

This file documents how one photo (or a batch / video of a stretch) becomes a
capacity, V/C and Level-of-Service result. Every result JSON also carries an
`inputs_and_sources` list saying, for that run, what was **measured**,
**entered**, **IRC-sourced** or set by **default**.

## Standards used

| Code | Used for |
|---|---|
| IRC:106-1990, Table 2 | Design Service Volume (DSV), urban roads: the design demand |
| IRC:106-1990, Table 1 | PCU factors by each vehicle type's own share (5 % / ≥10 %) |
| IRC:106-1990, cl. 5.3 | Average travel speed at each LOS as a share of free-flow speed |
| IRC:106 (2022 **draft** revision) Table 5, Eq. 5, Eq. 6, Tables 3, 8, 9, 12–16 | Lane capacity from operating speed; side-friction factors; LOS by V/C; operating speeds |
| IRC:82-2015, cl. 7.5.3, Table 5.3 | Pothole classes, repair method, urban condition rating |
| IRC:81-1997 / IRC:115-2014 | Structural evaluation when pothole condition is "Poor" |
| IRC:SP:55-2014, IRC:67-2012, IRC:35 | Work-zone management, signs, markings |

IRC:SP:83 (cited in earlier versions) covers cement-concrete pavements and is
no longer used for bituminous potholes.

## Formulas

1. **Geometry.** The 4 carriageway corners (near-left, near-right, far-right,
   far-left) define a homography *H* that maps image pixels to road-plane
   metres. Standing objects use the bottom edge of their box (ground
   contact); potholes use all four corners. Objects outside the carriageway
   are ignored. The corners come from:
   - the user's marks, when given; otherwise
   - **automatic edge detection** (`road_edges.py`):
     1. The asphalt colour is sampled at the bottom centre of the photo.
     2. Connected pixels within a lightness/colour tolerance form the road
        region.
     3. Each row's left and right boundary is extended across a painted edge
        line (IRC:35) when asphalt lies beyond it; a kerb is not crossed.
     4. A robust (RANSAC) straight line is fitted to each side, so parked
        vehicles and vendors cutting into the edge do not bend it.
     5. The near line is the bottom of the road; the far line is 75 % of the
        way to the vanishing point.

     The detection is rejected, with a reason, when the surface does not look
     like a road or covers less than 40 % of the photo width at the bottom.
   - If neither is available, `x_m = x_px × W / image_width` is used, and the
     result is flagged.

   The real scale always comes from the carriageway width *W* entered by the
   user. In tests on synthetic perspective roads (with and without edge lines,
   shadows, parked vehicles, an off-centre camera, a footpath close to asphalt
   colour), the detected edges were within 7.5 cm of the true edges on a 7 m
   carriageway. Straight roads only.
2. **Capacity.** Read from draft IRC:106 (2022) Table 5, capacity per lane:

   | Road type (our selection) | Table 5 row | PCU/h per lane |
   | --- | --- | --- |
   | 2-lane two-way | Two-lane undivided | 1,100 |
   | 4-lane undivided, 6-lane undivided | Four-lane undivided | 1,250 |
   | 4/6/8-lane divided, 2/3-lane one-way | Divided | 1,700 |

   Capacity C0 = lanes in the photo × the per-lane value. Six-lane undivided
   and one-way roads are not listed in Table 5 and take the nearest listed
   row. Because Table 5 is a base capacity with no side friction, every
   side-friction factor (F_OP, F_BS, F_AC, F_PD) applies on all fringe types.

3. **Lane-by-lane geometric capacity.** `C_geom = Σ (C0/n) × u_i × p_i`
   - `u_i`: usable free width in lane *i* ÷ lane width. Free gaps narrower than
     `min_usable_gap_m` (default 1.0 m) count as unusable.
   - `p_i`: pothole factor of the worst pothole touching lane *i*.
4. **Pothole factor.** `p = (20.11·V_pothole + 210.8) / (20.11·V_normal + 210.8)`
   (draft IRC:106 Eq. 5). V_pothole is measured, or defaults to 90/80/60 % of
   V_normal for IRC:82 small/medium/large.
5. **Side friction.** `C_eff = C_geom × F_OP × F_BS × F_AC × F_PD`
   (draft IRC:106 Eq. 6, Tables 12–16).
6. **V/C and LOS.** `V/C = demand / C_eff`. Demand is the observed volume, or
   the DSV if none is entered. LOS comes from draft IRC:106 Table 8
   (undivided) or Table 9 (divided / one-way).
7. **Speed.** `v = V_operating × fraction(LOS)`, with fractions
   A 0.90, B 0.70, C 0.50, D 0.40, E 0.33 and F 0.29 (IRC:106-1990 cl. 5.3;
   F = midpoint of 25–33 %).
8. **Vehicles/h.** `veh/h = PCU/h ÷ Σ p_i·PCU_i`. Each PCU_i comes from
   IRC:106 Table 1 at type *i*'s own share, interpolated between 5 % and 10 %.
9. **Priorities.**
   - The greedy restoration plan is the smallest ordered set of removals that
     brings the road back to LOS C.
   - Defects in that plan are URGENT at LOS E/F and ROUTINE at LOS D; all
     others are MONITOR.
   - IRC:82 large potholes are always URGENT.
10. **Segment (batch/video).** Segment capacity is the minimum over all samples
    (bottleneck). Parking occupancy is the share of samples showing parking.

## Status of the 38 earlier items

| # | Original code | Now |
|---|---|---|
| 1 | Photo edges = carriageway edges | Automatic road-edge detection or 4 marked corners → homography; fallback flagged |
| 2 | No perspective | Homography |
| 3 | Box height as length | Removed; length from homography when the near–far distance is given |
| 4 | Fixed downscale / imgsz | Keeps ≥ 2×imgsz; `ROAD_IMGSZ` configurable |
| 5 | One photo = segment | Batch/video segment result (bottleneck) |
| 6 | 40 m RoadRunner segment | Calibrated length or photo spacing; measured along-road positions |
| 7 | Guessed thresholds | `tools/calibrate_thresholds.py` (F1-optimal on your validation set); defaults flagged until run |
| 8 | Vendor/cart always vetoed | Keeps the more confident label |
| 9 | Vehicles = moving | Kept and flagged: a single photo cannot show motion |
| 10 | tree_on_road listed | Classes read from the model |
| 11 | DSV = capacity | Capacity from IRC:106 (2022) Table 5; DSV used as design demand; V/C and LOS reported |
| 12 | Linear width factor | Lane-by-lane usable width; unusable slivers |
| 13 | Full-length blockage | Bottleneck principle; parking by occupancy |
| 14 | Multiplicative factors | Draft IRC:106 Eq. 6 |
| 15 | Parking as width | F_OP (Table 12); no double count with collector DSV |
| 16 | Bus stops, access, pedestrians ignored | F_BS, F_AC, F_PD |
| 17 | Lanes/shoulder unused | Lanes drive capacity; shoulder reported only (IRC capacity is carriageway) |
| 18 | Fringe trusted | Factors skipped where already included; warning if parking contradicts |
| 19 | Pothole bbox-area bands | IRC:82 class by measured width |
| 20 | Fixed 2/5/9 cm depths | "Not measured" unless entered; entered depth used in the class |
| 21 | 0.95/0.85/0.70 penalties | Eq. 5 speed ratio; measured speed input |
| 22 | Worst pothole only | Per-lane factors; count and area % used in rectification |
| 23 | MiDaS constants | MiDaS removed from the calculation (no mm depth from one photo) |
| 24 | Fixed composition | Classified-count input; default flagged |
| 25 | 15 % heavy-vehicle regime | Table 1 by each type's own share |
| 26 | Wrong PCU values | Table 1 values |
| 27 | Average PCU floor 0.5 | Removed; shares validated and normalised |
| 28 | Free-flow speed table | Draft Table 3 or measured |
| 29 | Greenshields speed (rose with damage) | LOS-based speed (IRC:106 cl. 5.3) |
| 30 | Jam density | Removed |
| 31 | Twin ramp | Display only, labelled |
| 32 | MATLAB twin constants | Visualisation only; no effect on any number |
| 33 | % loss bands | LOS-based guidance |
| 34 | 5/15 % tiers | Restoration plan to LOS C + IRC:82 safety override |
| 35 | Escalation triggers | IRC:82 class and Table 5.3 "Poor" → IRC:81/115 evaluation (≥5 potholes kept as a cluster rule) |
| 36 | Repair deadlines | `PRIORITY_POLICY`: department policy, labelled, not IRC |
| 37 | Time-based sampling, box IoU tracking | Distance-based sampling option; tracking on lateral metres |
| 38 | Batch worst/mean only | Bottleneck segment capacity, LOS, measured parking occupancy |

## Default values (always listed in the result)

- **Narrowest usable gap:** 1.0 m (adjustable).
- **Speed at potholes:** 90/80/60 % of the operating speed when not measured.
- **Unknown parking occupancy:** the lowest F_OP band (0.70) when parking is
  seen in a single photo.
- **Mixed editions:** the IRC:106-1990 DSV is combined with the IRC:106 (2022)
  **draft** factors and LOS tables. Check whether the final revision has been
  published.
- **Divided roads:** the Table 2 DSV is treated as a two-way total and halved
  for one direction.
- **Eq. 5:** stated for divided roads; it is used as a ratio for all roads.
- **Parking occupancy from samples:** needs at least 3 evenly spaced photos or
  frames.
- **Pothole depth:** cannot be measured from a photo, so IRC:82 depth limits
  need a field check unless a depth is entered.
