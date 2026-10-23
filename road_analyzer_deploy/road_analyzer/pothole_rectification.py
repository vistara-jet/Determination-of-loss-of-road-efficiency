"""
pothole_rectification.py
--------------------------
Turns the per_defect['pothole'] output of core.py (IRC:82 class, measured
width, optional measured depth, pothole area %, capacity impact) into a
rectification measure.

Reference standards
  IRC:82-2015   Code of Practice for Maintenance of Bituminous Roads
                - cl. 7.5.3.4 pothole classes (small / medium / large)
                - cl. 7.5.3.5 rectification: filling with open- or dense-graded
                  premix (hot/cold mixes, storable cold mixes per IRC:116 /
                  IRC:100, ready-made mixes, patching machines)
                - Table 5.3 urban condition rating from pothole area %
  IRC:81-1997 / IRC:115-2014  structural evaluation (Benkelman beam / FWD)
                when the condition rating is Poor or failure is clustered
  IRC:SP:55-2014  traffic management in work zones; IRC:67-2012 signs

Note: IRC:SP:83 (cited previously) covers CEMENT CONCRETE pavements and
does not apply to potholes in bituminous roads.

Repair priorities (days/hours) are a department policy, not an IRC value;
they are kept in PRIORITY_POLICY so a department can set its own.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

PRIORITY_POLICY = {
    "large":  "Emergency (48 hr): road-user safety hazard",
    "medium": "Urgent (7 days)",
    "small":  "Routine (30 days)",
}


@dataclass
class RectificationMeasure:
    category: str
    method: str
    materials: str
    irc_reference: str
    priority: str
    work_zone_note: str
    escalation_note: str = ""


def recommend_pothole_rectification(worst_class: str, count: int,
                                     condition_rating: Optional[str] = None,
                                     measured_depth_mm: Optional[float] = None) -> RectificationMeasure:
    """worst_class: IRC:82 'small' | 'medium' | 'large' (core.py)."""
    if worst_class not in PRIORITY_POLICY:
        worst_class = {"shallow": "small", "moderate": "medium", "deep": "large"}.get(worst_class, "small")

    structural = condition_rating == "Poor" or count >= 5
    escalation = ""
    if structural:
        escalation = ("Pothole area exceeds 0.5 % of the road (IRC:82 Table 5.3 'Poor') or failures are "
                      "clustered: carry out structural evaluation (Benkelman beam per IRC:81-1997 or FWD "
                      "per IRC:115-2014) before choosing patching versus renewal/overlay.")

    if worst_class == "large":
        return RectificationMeasure(
            category="Deep patch (pothole extending into base course)",
            method=("Cut the pothole to a rectangular shape with vertical sides down to sound material, "
                    "remove loose and wet material, rebuild any failed base course in compacted layers, "
                    "apply tack coat, fill with dense-graded premix and compact flush with the surface "
                    "(IRC:82 cl. 7.5.3.5)."),
            materials="Granular base material as needed; tack coat; dense-graded bituminous premix (hot mix, "
                      "or storable cold mix per IRC:116 / IRC:100 in wet weather).",
            irc_reference="IRC:82-2015 cl. 7.5.3.4 (large: > 50 mm deep, > 500 mm wide) and cl. 7.5.3.5",
            priority=PRIORITY_POLICY["large"],
            work_zone_note="Lane closure with cones, barricades and signs per IRC:SP:55-2014 / IRC:67-2012.",
            escalation_note=escalation or ("Measured depth needed to confirm base-course involvement."
                                           if measured_depth_mm is None else ""),
        )
    if worst_class == "medium":
        return RectificationMeasure(
            category="Premix patch",
            method=("Trim to a regular shape, clean and dry, apply tack coat, fill with open- or "
                    "dense-graded premix and compact flush (IRC:82 cl. 7.5.3.5)."),
            materials="Tack coat (bitumen emulsion) + open- or dense-graded premix; cold mix per "
                      "IRC:116 / IRC:100 where hot mix is not available.",
            irc_reference="IRC:82-2015 cl. 7.5.3.4 (medium: 25-50 mm deep, ~500 mm wide) and cl. 7.5.3.5",
            priority=PRIORITY_POLICY["medium"],
            work_zone_note="Short lane closure with cones and warning signs per IRC:SP:55-2014.",
            escalation_note=escalation,
        )
    return RectificationMeasure(
        category="Premix filling (small pothole)",
        method="Clean out, apply tack coat, fill with premix or ready-made cold mix and compact (IRC:82 cl. 7.5.3.5).",
        materials="Ready-made / storable cold mix or open-graded premix.",
        irc_reference="IRC:82-2015 cl. 7.5.3.4 (small: ~25 mm deep, ~200 mm wide) and cl. 7.5.3.5",
        priority=PRIORITY_POLICY["small"],
        work_zone_note="Short-duration caution signs (IRC:67-2012); no closure normally needed.",
        escalation_note=escalation,
    )


def build_pwd_report_row(pothole_defect_entry: dict, location: str) -> dict:
    depth = pothole_defect_entry.get("depth_summary", {}) or {}
    rec = recommend_pothole_rectification(
        worst_class=depth.get("worst_severity", "small"),
        count=pothole_defect_entry.get("count", 1),
        condition_rating=depth.get("irc82_condition_rating"),
        measured_depth_mm=depth.get("measured_depth_mm"),
    )
    return {
        "location": location,
        "pothole_count": pothole_defect_entry.get("count"),
        "irc82_class": depth.get("worst_severity"),
        "worst_width_m": depth.get("worst_width_m"),
        "measured_depth_mm": depth.get("measured_depth_mm"),
        "avg_depth_cm": depth.get("avg_estimated_depth_cm"),
        "irc82_condition_rating": depth.get("irc82_condition_rating"),
        "capacity_loss_pct": pothole_defect_entry.get("capacity_loss_pct"),
        "rectification_category": rec.category,
        "method": rec.method,
        "materials": rec.materials,
        "irc_reference": rec.irc_reference,
        "priority": rec.priority,
        "work_zone_note": rec.work_zone_note,
        "escalation_note": rec.escalation_note,
    }
