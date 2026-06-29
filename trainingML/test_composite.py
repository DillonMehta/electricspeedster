#!/usr/bin/env python3
"""
test_composite.py — the headline composite (cuebench_signals.composite) and the
Description/specificity blend (cuebench_signals.blend_description).

The score is the mean of the four axes; weights are currently EVEN (all 0.25).

Run:  python3 -m pytest test_composite.py -q
"""
from __future__ import annotations
import cuebench_signals as sig


def test_weights_match_chosen_scheme():
    assert sig.AXIS_WEIGHTS == {"delegation": 0.25, "description": 0.25,
                                "discernment": 0.25, "diligence": 0.25}


def test_even_weights_equal_plain_mean():
    v = {"delegation": 80, "description": 80, "discernment": 50, "diligence": 50}
    assert abs(sig.composite(v) - 65.0) < 1e-6        # even weights -> plain average
    v2 = {"delegation": 70, "description": 60, "discernment": 90, "diligence": 40}
    assert abs(sig.composite(v2) - sum(v2.values()) / 4.0) < 1e-6


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
