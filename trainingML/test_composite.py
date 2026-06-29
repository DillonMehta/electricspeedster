#!/usr/bin/env python3
"""
test_composite.py — the WEIGHTED headline composite (cuebench_signals.composite) and the
Description/specificity blend (cuebench_signals.blend_description).

The score is a weighted mean of the four axes (outcome-heavy: discernment + diligence carry
0.30 each, delegation + description 0.20 each), not an even average.

Run:  python3 -m pytest test_composite.py -q
"""
from __future__ import annotations
import cuebench_signals as sig


def test_weights_match_chosen_scheme():
    assert sig.AXIS_WEIGHTS == {"delegation": 0.20, "description": 0.20,
                                "discernment": 0.30, "diligence": 0.30}


def test_weighted_mean_value():
    v = {"delegation": 80, "description": 80, "discernment": 50, "diligence": 50}
    # 0.2*80 + 0.2*80 + 0.3*50 + 0.3*50 = 32 + 30 = 62  (even mean would be 65)
    assert abs(sig.composite(v) - 62.0) < 1e-6


def test_outcome_strength_beats_even_average():
    """A session strong on the outcome axes (discernment/diligence) but weak on the input
    axes scores HIGHER than an even average — that's the point of the reweighting."""
    v = {"delegation": 40, "description": 40, "discernment": 90, "diligence": 90}
    even = sum(v.values()) / 4.0
    assert sig.composite(v) > even          # 70 vs 65


def test_outcome_weakness_costs_more():
    """Weakness on the HEAVIER outcome axes drags the score down more than an even mean would."""
    v = {"delegation": 70, "description": 70, "discernment": 70, "diligence": 30}
    # weighted = 0.2*70+0.2*70+0.3*70+0.3*30 = 14+14+21+9 = 58 ; even = 60
    assert abs(sig.composite(v) - 58.0) < 1e-6


def test_blend_description_60_40():
    assert sig.blend_description(80, 30) == 60        # 0.6*80 + 0.4*30 = 48 + 12
    assert sig.blend_description(100, 0) == 60
    assert sig.blend_description(50, 50) == 50


def test_blend_description_none_specificity_unchanged():
    assert sig.blend_description(73, None) == 73      # encoder unavailable -> description as-is


def test_missing_axis_is_renormalized_not_deflated():
    """If an axis is absent, weights renormalize over what's present (no silent deflation)."""
    v = {"delegation": 80, "description": 80}      # both 80 -> 80, not 80*0.6
    assert abs(sig.composite(v) - 80.0) < 1e-6


def test_fallback_to_even_mean_when_no_weighted_axis():
    assert abs(sig.composite({"mystery": 50, "other": 70}) - 60.0) < 1e-6
    assert sig.composite({}) == 0.0
