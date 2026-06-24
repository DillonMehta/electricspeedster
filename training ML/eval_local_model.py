#!/usr/bin/env python3
"""
CueBench — eval_local_model: how well does the local model reproduce the judge?
==============================================================================
Runs the trained DeBERTa model on the HELD-OUT test set (never seen in training)
and reports per-axis Spearman rho vs the judge scores. This is the proof.

  python eval_local_model.py --data data/ --model cuebench_model/
"""
from __future__ import annotations
import argparse, json
import numpy as np, torch
from transformers import AutoTokenizer
from scipy.stats import spearmanr
from train_cuebench import Regressor, AXES

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--model", default="cuebench_model")
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    import os
    tok = AutoTokenizer.from_pretrained(a.model)
    model = Regressor()
    bin_path = f"{a.model}/pytorch_model.bin"
    st_path = f"{a.model}/model.safetensors"
    if os.path.exists(bin_path):
        sd = torch.load(bin_path, map_location="cpu")
        loaded = "pytorch_model.bin"
    elif os.path.exists(st_path):
        from safetensors.torch import load_file
        sd = load_file(st_path)
        loaded = "model.safetensors"
    else:
        raise FileNotFoundError(f"No weights found in {a.model} (looked for pytorch_model.bin and model.safetensors)")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded weights from {loaded}  (missing={len(missing)}, unexpected={len(unexpected)})")
    if len(missing) > 5:
        print(f"  WARNING: many missing keys — weights may not have loaded correctly. sample: {missing[:5]}")
    model.to(dev).eval()

    rows = [json.loads(l) for l in open(f"{a.data}/test.jsonl")]
    preds = np.zeros((len(rows),4)); labels = np.array([r["labels"] for r in rows]); mask = np.array([r["mask"] for r in rows])
    with torch.no_grad():
        for i, r in enumerate(rows):
            enc = tok(r["text"], truncation=True, max_length=512, return_tensors="pt").to(dev)
            preds[i] = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])["pred"][0].cpu().numpy()

    print(f"\nHELD-OUT test set: {len(rows)} sessions")
    print("="*56)
    print("local model vs judge (per axis):")
    for i, ax in enumerate(AXES):
        m = mask[:,i]==1
        if m.sum()<10: print(f"  {ax:12} too few"); continue
        rho = spearmanr(preds[m,i], labels[m,i]).correlation
        mae = np.abs(preds[m,i]-labels[m,i]).mean()*100
        flag = "  STRONG" if rho>=.7 else ("  ok" if rho>=.5 else "  WEAK")
        print(f"  {ax:12} rho={rho:+.3f}  MAE={mae:.1f}pts  (n={int(m.sum())}){flag}")
    print("="*56)
    print("rho>=0.7 = local model faithfully reproduces the judge on that axis.")
    print("0.5-0.7 = usable lossy copy. <0.5 = the small model can't capture it.")

if __name__ == "__main__":
    main()
