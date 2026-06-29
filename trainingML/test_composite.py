#!/usr/bin/env python3
"""
test_composite.py — the WEIGHTED headline composite (cuebench_signals.composite).

The score is a weighted mean of the four axes ("driving skill heavier": delegation +
description carry 0.30 each, discernment + diligence 0.20 each), not an even average.

Run:  python3 -m pytest test_composite.py -q
"""
from __future__ import annotations
import cuebench_signals as sig


def test_weights_match_chosen_scheme():
    assert sig.AXIS_WEIGHTS == {"delegation": 0.30, "description": 0.30,
                                "discernment": 0.20, "diligence": 0.20}


def test_weighted_mean_value():
    v = {"delegation": 80, "description": 80, "discernment": 50, "diligence": 50}
    # 0.3*80 + 0.3*80 + 0.2*50 + 0.2*50 = 48 + 20 = 68  (even mean would be 65)
    assert abs(sig.composite(v) - 68.0) < 1e-6


def test_driving_strength_beats_even_average():
    """A session strong on the driving axes but weak on outcome axes scores HIGHER than
    an even average would give (that's the point of the reweighting)."""
    v = {"delegation": 90, "description": 90, "discernment": 40, "diligence": 40}
    even = sum(v.values()) / 4.0
    assert sig.composite(v) > even          # 71 vs 65


def test_outcome_weakness_costs_less_than_before():
    """Symmetrically, weakness on the lighter axes drags the score down less."""
    v = {"delegation": 70, "description": 70, "discernment": 70, "diligence": 30}
    # weighted = 0.3*70+0.3*70+0.2*70+0.2*30 = 42+14+6 = 62 ; even = 60
    assert abs(sig.composite(v) - 62.0) < 1e-6


def test_missing_axis_is_renormalized_not_deflated():
    """If an axis is absent, weights renormalize over what's present (no silent deflation)."""
    v = {"delegation": 80, "description": 80}      # both 80 -> 80, not 80*0.6
    assert abs(sig.composite(v) - 80.0) < 1e-6


def test_fallback_to_even_mean_when_no_weighted_axis():
    assert abs(sig.composite({"mystery": 50, "other": 70}) - 60.0) < 1e-6
    assert sig.composite({}) == 0.0
