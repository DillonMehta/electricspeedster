#!/usr/bin/env python3
"""
test_calibrate.py — the v1 length-bias decorrelation (cuebench_calibrate.decorrelate).

Verifies the correction does what it claims: centered (avg-length session unchanged), pulls
long sessions DOWN and short sessions UP, hits the biased axes (discernment/diligence) harder
than the clean ones (delegation), clips to [0,100], and passes unknown axes through.

Run:  python3 -m pytest test_calibrate.py -q
"""
from __future__ import annotations
import math
import cuebench_calibrate as calib

# support whose log1p == L0 -> the centering point (correction == 0 there)
MID_SUPPORT = round(math.exp(calib.L0) - 1)     # ~55


def test_center_is_unchanged():
    v = {"delegation": 70, "description": 65, "discernment": 60, "diligence": 55}
    out = calib.decorrelate(v, MID_SUPPORT)
    for ax in v:
        assert abs(out[ax] - v[ax]) <= 1            # at the corpus mean length, ~no change


def test_long_sessions_pulled_down():
    v = {"delegation": 75, "description": 75, "discernment": 75, "diligence": 75}
    out = calib.decorrelate(v, 500)                  # long session
    for ax in v:
        assert out[ax] <= v[ax]
    # biased axes (big BETA) drop more than the clean axis (small BETA)
    assert (v["discernment"] - out["discernment"]) > (v["delegation"] - out["delegation"])
    assert (v["diligence"] - out["diligence"]) > (v["delegation"] - out["delegation"])


def test_short_sessions_nudged_up():
    v = {"delegation": 50, "description": 50, "discernment": 50, "diligence": 50}
    out = calib.decorrelate(v, 12)                   # short (but substantive) session
    assert out["discernment"] >= v["discernment"]
    assert out["diligence"] >= v["diligence"]
    # and the biased axes move more than the clean one
    assert (out["diligence"] - v["diligence"]) >= (out["delegation"] - v["delegation"])


def test_clipped_to_unit_range():
    assert calib.decorrelate({"diligence": 98}, 5000)["diligence"] >= 0
    assert calib.decorrelate({"diligence": 99}, 1)["diligence"] <= 100
    out = calib.decorrelate({"discernment": 2}, 5000)
    assert 0 <= out["discernment"] <= 100


def test_unknown_axis_passes_through():
    out = calib.decorrelate({"mystery": 42}, 500)    # not in BETA -> unchanged
    assert out["mystery"] == 42


def test_returns_ints():
    out = calib.decorrelate({"delegation": 70, "diligence": 60}, 300)
    assert all(isinstance(x, int) for x in out.values())
