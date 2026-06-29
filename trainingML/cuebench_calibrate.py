#!/usr/bin/env python3
"""
cuebench_calibrate.py — length-bias decorrelation for the 4D scorer (v1 calibration)
====================================================================================
WHY: the local model (and, more mildly, the gpt-5.5 judge it distilled) scores longer
sessions higher — Spearman(session length, score) was ~+0.78 for the model vs only ~+0.37
for the judge: the model roughly DOUBLED the judge's length bias, latching onto raw tool/edit
counts and prompt verbosity as an easy quality proxy. The damage concentrates in `discernment`
(length misread as care) and `diligence` (volume misread as thoroughness); `delegation` /
`description` are barely affected.

WHAT: a closed-form, per-axis correction applied to the model's output BEFORE compositing/zoning:

    corrected_axis = clip( pred_axis - C * BETA[axis] * (log1p(support) - L0),  0, 100 )

`support` = operator prompts + tool calls (cuebench_signals.n_effective). The term is CENTERED
on the corpus-mean log-support (L0), so the AVERAGE session is unchanged: long sessions are
pulled down and short ones nudged up, each axis in proportion to its measured length slope
(BETA). This removes most of the model's EXCESS length amplification while keeping a little
signal (target composite Spearman(support) ~+0.18, "halfway") rather than zeroing it.

CONSTANTS were fit by scratchpad/fit_calib.py on the 5,781-session training corpus (the model's
reference population): per-axis OLS slope of the model's prediction vs centered log-support, then
a sweep of C to hit the target. Mean score level is preserved (57.0 -> 57.0 on the sample).

This is a v1 OUTPUT calibration — a stopgap until the model is retrained with rate-based
features (verify/tool, loops/tool, churn/edit, revert rate) that remove the bias at the source.
Re-fit (rerun fit_calib.py) whenever the model is retrained, and consider re-running
cuebench_calibrate_zones.py since the score distribution shifts slightly under this transform.
"""
from __future__ import annotations
import math

# --- fitted on the training corpus; see module docstring / scratchpad/fit_calib.py ---
L0 = 4.0254                                   # mean log1p(support) over the corpus (centering point)
BETA = {                                      # model's prediction slope vs centered log-support, per axis
    "delegation": 2.2944,
    "description": 3.6177,
    "discernment": 8.6210,
    "diligence": 8.1543,
}
C = 0.65                                      # decorrelation strength: composite Spearman(support) ~+0.18
TARGET_SPEARMAN = 0.18                         # documented intent for this C

# Clamp the centering term so a pathologically short/long session can't get an extreme swing.
# Range covers ~support 5 (Lc≈-2.2) to ~support 3700 (Lc≈+4.2) seen in the corpus.
_LC_MIN, _LC_MAX = -2.2, 4.3


def decorrelate(vectors: dict, support: int) -> dict:
    """Apply the length-bias correction to a {axis: 0-100} vector dict.

    Returns a NEW dict of ints in [0,100]. `support` is prompts + tool calls
    (cuebench_signals.n_effective). Unknown axes (not in BETA) pass through unchanged.
    The transform is centered, so a session at the corpus-mean length is unchanged."""
    lc = math.log1p(max(0, int(support or 0))) - L0
    lc = max(_LC_MIN, min(_LC_MAX, lc))
    out = {}
    for ax, v in vectors.items():
        adj = C * BETA.get(ax, 0.0) * lc
        out[ax] = int(round(min(100.0, max(0.0, float(v) - adj))))
    return out
