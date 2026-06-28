#!/usr/bin/env python3
"""
CueBench — train_cuebench (DeBERTa-v3-small, 4D regression, masked loss)
=======================================================================
Distills the gpt-5.5 judge into a small local model. 4 regression heads
(del/desc/disc/dil), each in 0-1. Masked MSE so N/A axes don't train on a fake 0.

  pip install "torch>=2.4" transformers datasets accelerate
  python train_cuebench.py --data data/ --out cuebench_model/ --epochs 4

Runs on Apple MPS, CUDA, or CPU (CPU is slow but works for this size).
"""
from __future__ import annotations
import argparse, json
import numpy as np, torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModel, Trainer, TrainingArguments)
import torch.nn as nn

MODEL_NAME = "roberta-base"   # switched from deberta-v3-small: disentangled attn crashes on MPS (Failure Mode #2)
AXES = ["delegation","description","discernment","diligence"]

class DS(Dataset):
    def __init__(self, path, tok, maxlen=512):
        self.rows = [json.loads(l) for l in open(path)]
        self.tok = tok; self.maxlen = maxlen
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        # No padding here — DynamicCollator pads each batch to its own longest seq.
        # Fixed padding to maxlen wasted huge compute/memory (median ~346 tok, 64% < 512).
        enc = self.tok(r["text"], truncation=True, max_length=self.maxlen)
        return {"input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": r["labels"], "mask": r["mask"]}

class DynamicCollator:
    """Pad each batch to its longest sequence (capped at maxlen). Big speed/mem win
    over fixed-512 padding on MPS given the short-skewed token distribution."""
    def __init__(self, tok): self.tok = tok
    def __call__(self, batch):
        enc = self.tok.pad([{"input_ids": b["input_ids"],
                             "attention_mask": b["attention_mask"]} for b in batch],
                            padding=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.float),
                "mask": torch.tensor([b["mask"] for b in batch], dtype=torch.float)}

FREEZE_BOTTOM = 6   # freeze embeddings + bottom N of 12 layers (memory fix for swap-bound MPS)

class Regressor(nn.Module):
    def __init__(self, base=MODEL_NAME, n=4):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        # Swap-constrained MPS (17GB RAM, heavy swap): full fine-tune's ~2.5GB hot set
        # (grads + Adam states for all 125M params + activations) thrashes badly.
        # Freezing embeddings + bottom layers halves trainable params -> halves grad &
        # optimizer memory and frees frozen-layer activations, so the hot set fits in RAM.
        if FREEZE_BOTTOM > 0:
            for p in self.enc.embeddings.parameters():
                p.requires_grad = False
            for layer in self.enc.encoder.layer[:FREEZE_BOTTOM]:
                for p in layer.parameters():
                    p.requires_grad = False
        h = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, h), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(h, n), nn.Sigmoid())
    def forward(self, input_ids=None, attention_mask=None, labels=None, mask=None):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:,0]          # [CLS]
        pred = self.head(pooled)
        loss = None
        if labels is not None:
            se = (pred - labels)**2 * mask           # masked MSE
            loss = se.sum() / mask.sum().clamp(min=1)
        return {"loss": loss, "pred": pred}

def metrics_builder():
    from scipy.stats import spearmanr
    def compute(evalpred):
        preds, (labels, mask) = evalpred.predictions, evalpred.label_ids
        out = {}
        for i, ax in enumerate(AXES):
            m = mask[:,i] == 1
            if m.sum() > 10:
                rho = spearmanr(preds[m,i], labels[m,i]).correlation
                out[f"rho_{ax}"] = float(rho)
        out["rho_mean"] = float(np.mean(list(out.values()))) if out else 0.0
        return out
    return compute

class MaskTrainer(Trainer):
    # ensure labels+mask both reach compute_metrics
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        mask = inputs["mask"]; labels = inputs["labels"]
        with torch.no_grad():
            out = model(**inputs)
        return (out["loss"], out["pred"], (labels, mask))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="cuebench_model")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    train, val = DS(f"{a.data}/train.jsonl", tok), DS(f"{a.data}/val.jsonl", tok)
    model = Regressor()

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
        learning_rate=a.lr, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="rho_mean", greater_is_better=True,
        save_total_limit=2,
        logging_steps=50, report_to="none", warmup_ratio=0.1, weight_decay=0.01)

    trainer = MaskTrainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                          data_collator=DynamicCollator(tok), compute_metrics=metrics_builder())
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    print("\nval metrics:", trainer.evaluate())
    print(f"saved -> {a.out}")

if __name__ == "__main__":
    main()
