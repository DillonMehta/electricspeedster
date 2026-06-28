#!/usr/bin/env python3
"""
model_infer.py — load the trained CueBench 4D scorer once and score digests.
============================================================================
Loads cuebench_model/ (RoBERTa via train_cuebench.Regressor, safetensors) on CPU
and turns a digest string (from cuebench_signals.digest_text) into 4 axis scores.

  from model_infer import ModelScorer
  scorer = ModelScorer("cuebench_model")          # load once at startup
  vectors = scorer.score(digest)                  # -> {"delegation":int, ...} 0-100

CPU by design (plan gotcha): one session scores in <1s, no MPS quirks at inference.
"""
from __future__ import annotations
import os

# Quiet transformers' load-report chatter so the agent's stdout stays clean.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch
from transformers import AutoTokenizer

from train_cuebench import Regressor, AXES  # model definition + axis order (the contract)


class ModelScorer:
    def __init__(self, model_dir: str = "cuebench_model", device: str = "cpu", maxlen: int = 512):
        self.device = device
        self.maxlen = maxlen
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = Regressor()

        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        st_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(bin_path):
            sd = torch.load(bin_path, map_location="cpu")
            src = "pytorch_model.bin"
        elif os.path.exists(st_path):
            from safetensors.torch import load_file
            sd = load_file(st_path)
            src = "model.safetensors"
        else:
            raise FileNotFoundError(
                f"No weights in {model_dir} (looked for pytorch_model.bin and model.safetensors)")

        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if len(missing) > 5:
            raise RuntimeError(
                f"{len(missing)} missing keys loading {src} — weights likely didn't load "
                f"(sample: {missing[:5]}). Refusing to score with a half-initialized model.")
        self.model.to(device).eval()
        self.loaded_from = src

    @torch.no_grad()
    def score(self, digest: str) -> dict:
        """digest -> {axis: int 0-100}. Dynamic padding (single example, no fixed pad)."""
        enc = self.tok(digest, truncation=True, max_length=self.maxlen, return_tensors="pt").to(self.device)
        pred = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])["pred"][0]
        vals = (pred.cpu().numpy() * 100.0)
        return {ax: int(round(float(v))) for ax, v in zip(AXES, vals)}


if __name__ == "__main__":
    # Smoke test: score a hand-built digest so you can eyeball it before wiring the agent.
    import argparse, json
    from cuebench_signals import build_inputs, digest_text
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cuebench_model")
    a = ap.parse_args()
    s = ModelScorer(a.model)
    demo = build_inputs(
        prompts=["add a 300ms debounce to the search handler in search.js",
                 "now add a unit test for the empty-query case"],
        tool_cmds=["npm test", "git commit -m 'debounce'"],
        n_edits=2, loops=0,
        git={"n_commits": 1, "churn": 40, "reverts": 0, "survival_proxy": 1.0})
    dg = digest_text(demo)
    print("loaded_from:", s.loaded_from)
    print("digest:\n" + dg)
    print("vectors:", json.dumps(s.score(dg)))
