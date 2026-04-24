#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPS scoring baseline (extracted from ``PPS_1.py``).

All constants and formulas are copied from the current Streamlit prototype so
that **one-question-per-screen** apps and the legacy one-page app stay aligned.

Confidence **labels** (Ambiguous / Moderate / High) use the same thresholds as
``PPS_Scoring.py``, via ``confidence_label_from_margin`` in
``pps_mixed_ambiguous_logic.py``. For the Streamlit app, apply that function to the
margin computed from ``eff_dists`` here so the closest style matches ``final_style``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd

from pps_mixed_ambiguous_logic import ConfidenceLabel, confidence_label_from_margin

PERMISSIVE_ITEMS = [
    "X3",
    "X10",
    "X13",
    "X14",
    "X34",
    "X40",
    "X44",
    "X61",
    "X64",
    "X71",
    "X72",
    "X60",
]

AUTHORITATIVE_ITEMS = [
    "X5",
    "X7",
    "X19",
    "X33",
    "X43",
    "X47",
    "X52",
    "X62",
    "X65",
    "X67",
    "X78",
    "X80",
]

AUTHORITARIAN_ITEMS = [
    "X4",
    "X11",
    "X18",
    "X24",
    "X30",
    "X32",
    "X35",
    "X38",
    "X39",
    "X48",
    "X70",
    "X76",
]

STYLE_ORDER = ["Authoritarian", "Authoritative", "Permissive"]

ITEM_TEXTS: Dict[str, str] = {
    "X32": "I reprimand my pet to improve behavior",
    "X30": "I carry out discipline after my pet misbehaves",
    "X76": "If I give a command, I believe my pet should obey me because I said so",
    "X18": "I scold or criticize my pet's behavior if it doesn't meet my expectations",
    "X24": "I employ an \"alpha\" or \"pack leader\" mentality with my pet",
    "X39": "When I teach my pet commands I expect him/her to obey me no matter what",
    "X38": "I use short, quick pulls on the leash, collar, or scruff if my pet pulls when I'm handling it",
    "X4": "I use a stern, loud voice when my pet misbehaves",
    "X70": "I use physical aids when I train my pet",
    "X48": "I yell or shout when my pet misbehaves",
    "X35": "I demand that my pet does things",
    "X11": "I tell my pet what to do",
    "X52": "I think about my pet's feelings",
    "X47": "I spend (or try to spend) a lot of quality time with my pet",
    "X33": "I think about my pet when I'm away from home",
    "X67": "I have warm and fun times with my pet",
    "X65": "I laugh and joke with my pet",
    "X78": "I believe my pet displays human emotions",
    "X80": "If I make a mistake with my pet, I feel bad and try to make it up to him/her",
    "X7": "I wish I could spend more time with my pet",
    "X19": "I play with my pet",
    "X62": "I celebrate my pet's birthday",
    "X43": "I can recognize individuals that my pet gets along with",
    "X5": "I show patience with my pet",
    "X44": "I ignore my pet when it chews or steals something it's not supposed to",
    "X71": "I am unsure about how to handle my pet's misbehavior",
    "X13": "I worry about being too harsh when correcting my pet's misbehavior",
    "X40": "I give into my pet when he/she causes a commotion about something",
    "X14": "I am afraid that disciplining my pet for misbehavior will cause him to like me less",
    "X72": "I give into my pet when my pet is stubborn",
    "X64": "I think about punishing my pet but don't actually do it",
    "X3": "I allow my pet to annoy other animals",
    "X61": "I worry about how my pet feels about me, especially whether my pet likes me or not",
    "X10": "I worry that I will overly restrict my pet if I am too demanding",
    "X34": "If I have guests over and my pet misbehaves, I refrain from correcting it",
    "X60": "I spoil my pet",
}

CENTROIDS = {
    "Authoritarian": np.array([1.864583, 4.083333, 3.156250]),
    "Authoritative": np.array([2.125000, 4.387821, 2.285256]),
    "Permissive": np.array([2.645833, 3.802083, 2.458333]),
}

BETA_PERM = 1.205
BETA_AUTHV = 1.000
BETA_AUTHN = 1.325

MEAN_PERM = 2.093652
SD_PERM = 0.516713
MEAN_AUTHV = 4.245453
SD_AUTHV = 0.530544
MEAN_AUTHN = 2.533141
SD_AUTHN = 0.730939

LIKERT_LABELS = [
    "Never",
    "Once in a while",
    "About half the time",
    "Very often",
    "Always",
]


def ordered_item_ids() -> List[str]:
    """Return X-item ids sorted by numeric suffix (same order as legacy UI)."""
    return sorted(ITEM_TEXTS.keys(), key=lambda x: int(x[1:]))


# Tie-break order when two styles share the same effective distance (matches
# ``min(eff_dists, key=eff_dists.get)`` iteration over the dict built below).
_EFF_DIST_TIE_ORDER = ["Permissive", "Authoritative", "Authoritarian"]


def styles_ranked_by_effective_distance(
    eff_dists: Dict[str, float],
) -> List[Tuple[str, float]]:
    """Return (style, effective_distance) pairs sorted best → worst."""
    tie_idx = {s: i for i, s in enumerate(_EFF_DIST_TIE_ORDER)}
    items = list(eff_dists.items())
    items.sort(key=lambda x: (x[1], tie_idx[x[0]]))
    return items


def margin_and_confidence_from_eff_dists(
    eff_dists: Dict[str, float],
) -> Tuple[float, ConfidenceLabel, str, str]:
    """Distance margin and label using the same effective distances as ``final_style``.

    Returns:
        ``(margin, label, best_style, second_style)`` where ``margin`` is
        ``round(d_second - d_best, 3)`` on biased effective distances.
    """
    ranked = styles_ranked_by_effective_distance(eff_dists)
    _best_s, best_d = ranked[0]
    _second_s, second_d = ranked[1]
    margin = round(float(second_d - best_d), 3)
    label = confidence_label_from_margin(margin)
    return margin, label, ranked[0][0], ranked[1][0]


def z_to_percentile(z: float) -> float:
    """Convert z-score to percentile (0–100) assuming normal distribution."""
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return 100.0 * cdf


@dataclass(frozen=True)
class PPSComputed:
    """Full prototype scoring snapshot (matches former inline ``PPS_1.py`` block)."""

    s_perm: float
    s_authv: float
    s_authn: float
    final_style: str
    z_perm: float
    z_authv: float
    z_authn: float
    p_perm: float
    p_authv: float
    p_authn: float
    eff_dists: Dict[str, float]
    sim_scores: Dict[str, float]


def likert_label_to_score(label: str) -> int:
    """Map a Likert label string to 1–5 (raises ValueError if unknown)."""
    return LIKERT_LABELS.index(label) + 1


def compute_from_likert_labels(responses: Dict[str, Optional[str]]) -> Optional[PPSComputed]:
    """Compute profile from raw radio labels; returns None if any answer missing."""
    if any(responses.get(i) is None for i in ordered_item_ids()):
        return None
    scores_1_5: Dict[str, int] = {}
    for item in ordered_item_ids():
        lbl = responses[item]
        if lbl is None:
            return None
        scores_1_5[item] = likert_label_to_score(str(lbl))
    return compute_from_numeric(scores_1_5)


def compute_from_numeric(responses_1_5: Dict[str, int]) -> PPSComputed:
    """Compute profile when each item is already coded 1–5."""
    def mean_for(items: List[str]) -> float:
        vals = [responses_1_5[i] for i in items]
        return float(np.mean(vals))

    s_perm = mean_for(PERMISSIVE_ITEMS)
    s_authv = mean_for(AUTHORITATIVE_ITEMS)
    s_authn = mean_for(AUTHORITARIAN_ITEMS)
    profile = np.array([s_perm, s_authv, s_authn])

    dists: Dict[str, float] = {}
    for style in STYLE_ORDER:
        dists[style] = float(np.linalg.norm(profile - CENTROIDS[style]))

    eff_dists = {
        "Permissive": BETA_PERM * dists["Permissive"],
        "Authoritative": BETA_AUTHV * dists["Authoritative"],
        "Authoritarian": BETA_AUTHN * dists["Authoritarian"],
    }
    final_style = min(eff_dists, key=eff_dists.get)

    z_perm = (s_perm - MEAN_PERM) / SD_PERM
    z_authv = (s_authv - MEAN_AUTHV) / SD_AUTHV
    z_authn = (s_authn - MEAN_AUTHN) / SD_AUTHN

    p_perm = z_to_percentile(z_perm)
    p_authv = z_to_percentile(z_authv)
    p_authn = z_to_percentile(z_authn)

    sim_scores = {
        style: float(np.exp(-eff_dists[style]))
        for style in STYLE_ORDER
    }

    return PPSComputed(
        s_perm=s_perm,
        s_authv=s_authv,
        s_authn=s_authn,
        final_style=final_style,
        z_perm=z_perm,
        z_authv=z_authv,
        z_authn=z_authn,
        p_perm=p_perm,
        p_authv=p_authv,
        p_authn=p_authn,
        eff_dists=eff_dists,
        sim_scores=sim_scores,
    )


def chart_percentile_bars(p: PPSComputed) -> alt.Chart:
    """Left-panel chart from prototype: normative percentiles by style."""
    df_pct = pd.DataFrame(
        {
            "Style": ["Authoritarian", "Authoritative", "Permissive"],
            "Percentile": [p.p_authn, p.p_authv, p.p_perm],
        }
    )
    color_scale = alt.Scale(
        domain=["Authoritarian", "Authoritative", "Permissive"],
        range=["#FF4B4B", "#2ECC71", "#F1C40F"],
    )
    return (
        alt.Chart(df_pct)
        .mark_bar()
        .encode(
            x=alt.X("Style:N", sort=list(STYLE_ORDER)),
            y=alt.Y("Percentile:Q", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Style:N", scale=color_scale, legend=None),
            tooltip=["Style", alt.Tooltip("Percentile:Q", format=".0f")],
        )
    )


def chart_similarity_donut(p: PPSComputed) -> alt.Chart:
    """Right-panel donut from prototype: exp(-effective distance)."""
    df_sim = pd.DataFrame(
        {
            "Style": list(p.sim_scores.keys()),
            "Similarity": list(p.sim_scores.values()),
        }
    )
    color_scale = alt.Scale(
        domain=["Authoritarian", "Authoritative", "Permissive"],
        range=["#FF4B4B", "#2ECC71", "#F1C40F"],
    )
    return (
        alt.Chart(df_sim)
        .mark_arc(innerRadius=50)
        .encode(
            theta="Similarity:Q",
            color=alt.Color("Style:N", scale=color_scale),
            tooltip=["Style", alt.Tooltip("Similarity:Q", format=".3f")],
        )
    )


def chart_subscale_means_simple(p: PPSComputed) -> alt.Chart:
    """Friendly bar chart of 1–5 subscale means (for low-clutter result screens)."""
    df = pd.DataFrame(
        {
            "Area": ["Authoritarian", "Authoritative", "Permissive"],
            "Average rating (1–5)": [p.s_authn, p.s_authv, p.s_perm],
        }
    )
    color_scale = alt.Scale(
        domain=["Authoritarian", "Authoritative", "Permissive"],
        range=["#FF4B4B", "#2ECC71", "#F1C40F"],
    )
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Area:N", sort=list(STYLE_ORDER)),
            y=alt.Y("Average rating (1–5):Q", scale=alt.Scale(domain=[0, 5])),
            color=alt.Color("Area:N", scale=color_scale, legend=None),
            tooltip=["Area", alt.Tooltip("Average rating (1–5):Q", format=".2f")],
        )
    )
