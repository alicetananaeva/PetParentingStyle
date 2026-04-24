#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPS mixed / ambiguous classification (extracted baseline).

This module isolates the **confidence / borderline** logic from the SfJ batch
scoring script ``PPS_Scoring.py`` (function ``classify_with_scientific_confidence``).
The Streamlit app combines **prototype centroids / betas** (``pps_scoring_baseline``)
with these **thresholds** and labels via ``confidence_label_from_margin`` applied
to the margin between the two smallest *effective* distances.

The batch helper ``classify_with_scientific_confidence`` still uses **rounded**
centroids for SfJ CSV pipelines; margins in the app are derived from the same
formula but from ``eff_dists`` in ``pps_scoring_baseline`` so the winner matches
``final_style``.

Threshold rationale:
- ``MARGIN_AMBIGUOUS_THRESHOLD`` = **1/24** of the parent distance scale (half of
  the former 1/12 per-item step) — if the margin between the two closest biased
  distances is *below* this, treat as **Ambiguous** and show mixed-style copy;
  at or *above* it, the secondary style is not shown as a mixed overlay.
- ``MARGIN_HIGH_CONFIDENCE_THRESHOLD`` (0.25) — approximate SEM-style separation
  for "high confidence" in a single style.

Constants are rounded to three decimals to match the batch pipeline; the
prototype centroids in ``PPS_1.py`` use full float literals — batch vs app
winners can differ slightly in edge cases; app margins always follow ``eff_dists``.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

# Centroids as in PPS_Scoring.py: [Permissive, Authoritative, Authoritarian]
CENTROIDS_SCORING = {
    "Authoritarian": np.array([1.865, 4.083, 3.156]),
    "Authoritative": np.array([2.125, 4.388, 2.285]),
    "Permissive": np.array([2.646, 3.802, 2.458]),
}

BETAS_SCORING = {"Permissive": 1.205, "Authoritative": 1.000, "Authoritarian": 1.325}

# Ambiguous / mixed: margin strictly below 1/24 → show secondary style (mixed copy).
MARGIN_AMBIGUOUS_THRESHOLD = 1.0 / 24.0
MARGIN_HIGH_CONFIDENCE_THRESHOLD = 0.25

ConfidenceLabel = Literal[
    "Ambiguous (Borderline)", "Moderate Confidence", "High Confidence"
]


def confidence_label_from_margin(margin: float) -> ConfidenceLabel:
    """Map distance margin (second − best effective distance) to a label.

    At or above ``MARGIN_AMBIGUOUS_THRESHOLD`` (1/24), the profile is *not* treated
    as borderline; secondary-style / mixed text is not shown for that case.
    """
    if margin < MARGIN_AMBIGUOUS_THRESHOLD:
        return "Ambiguous (Borderline)"
    if margin < MARGIN_HIGH_CONFIDENCE_THRESHOLD:
        return "Moderate Confidence"
    return "High Confidence"


def classify_with_scientific_confidence(
    perm: float, authv: float, authn: float
) -> Tuple[str, float, ConfidenceLabel]:
    """Classify parenting style with distance margin and confidence label.

    Args:
        perm: Mean Permissive subscale (1–5).
        authv: Mean Authoritative subscale (1–5).
        authn: Mean Authoritarian subscale (1–5).

    Returns:
        Tuple of ``(best_style, margin, confidence_label)`` where ``margin`` is
        ``round(second_best_effective_distance - best_effective_distance, 3)``.
    """
    target = np.array([perm, authv, authn], dtype=float)
    results: list[tuple[str, float]] = []

    for style, centroid in CENTROIDS_SCORING.items():
        dist = float(np.linalg.norm(target - centroid) * BETAS_SCORING[style])
        results.append((style, dist))

    results.sort(key=lambda x: x[1])

    best_style, best_dist = results[0]
    second_dist = results[1][1]
    margin = round(second_dist - best_dist, 3)
    label = confidence_label_from_margin(margin)

    return best_style, margin, label
