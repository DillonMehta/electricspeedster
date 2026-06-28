# CueBench — Local 4D Scorer: Training Results

**Date:** 2026-06-24
**Outcome:** ✅ Success. A small local model distilled from the gpt-5.5 judge reproduces
**all four** 4D axes at **STRONG fidelity (rho ≥ 0.7)** on a held-out test set — with **no API calls**.

---

## HEADLINE RESULT (held-out test set, 578 sessions — never seen in training)

Local model vs. gpt-5.5 judge, per axis (Spearman rho):

| Axis | rho | MAE (0–100 pts) | n | Verdict |
|---|---|---|---|---|
| **delegation**  | **+0.776** | 4.8 | 578 | STRONG (faithful) |
| **description** | **+0.851** | 5.9 | 578 | STRONG (faithful) |
| **discernment** | **+0.774** | 8.8 | 430 | STRONG (faithful) |
| **diligence**   | **+0.870** | 7.5 | 550 | STRONG (faithful) |
| **mean**        | **+0.818** | — | — | — |

Interpretation scale (from the handoff): rho ≥ 0.7 = faithful copy · 0.5–0.7 = usable lossy copy · < 0.5 = not captured.

**Per-axis verdict: all four axes are FAITHFUL (≥ 0.7).** None are merely lossy; none failed.
MAE is single-digit points on the 0–100 scale for every axis.

> `discernment` has n=430 (not 578) because the judge returned N/A for some sessions;
> the masked loss skips those, and eval correctly scores only the labeled subset.

---

## NORTH STAR — was it hit?

**Yes.** The premise was that only 2 of the 4Ds are cleanly measurable deterministically
(Description via embedding-specificity, Diligence via verification), while **Delegation and
Discernment are "latent"** — they require a model that reads the session text.

The local model recovers both latent axes at **STRONG** level on held-out data:
- **Delegation: 0.776** (was the open question → faithful)
- **Discernment: 0.774** (was the open question → faithful)

This unlocks the product: **a local, no-API, full-4D scorer.** It is the deliverable in `cuebench_model/`.

---

## WHAT PRODUCED IT

**Model:** `roberta-base` encoder + 2-layer MLP regression head (4 sigmoid outputs, one per axis).
- Switched from `microsoft/deberta-v3-small` — DeBERTa's disentangled attention crashes on MPS (handoff Failure Mode #2).
- **Bottom-half frozen:** embeddings + the bottom 6 of 12 transformer layers are frozen
  (43.7M of 125.2M params trainable, 35%). This was a memory fix (see below) — and it also
  **improved** generalization (less overfitting on a small/short-text dataset).

**Hyperparameters:**

| Setting | Value |
|---|---|
| epochs | 4 (best = epoch 4, selected by `load_best_model_at_end` on val rho_mean) |
| batch size | 8 |
| learning rate | 2e-5 (AdamW), warmup_ratio 0.1, weight_decay 0.01 |
| max sequence length | 512 tokens, **dynamic per-batch padding** (not fixed 512 — see below) |
| loss | masked MSE (per-axis mask skips N/A axes so they don't train toward a fake 0) |
| device | Apple MPS |
| train / val / test | 4625 / 578 / 578 sessions |
| trained model size | ~500 MB (`model.safetensors`) |
| training time | ~61 min for the full 4-epoch run |

**Axis coverage in labels:** delegation 5779, description 5781, discernment 4195, diligence 5561.

---

## VAL rho_mean CURVE ACROSS EPOCHS (full run, frozen model)

Monotonic improvement, no overfitting — every axis still climbing at epoch 4:

| epoch | delegation | description | discernment | diligence | **rho_mean** |
|---|---|---|---|---|---|
| 1 | 0.649 | 0.804 | 0.745 | 0.799 | 0.749 |
| 2 | 0.727 | 0.825 | 0.754 | 0.838 | 0.786 |
| 3 | 0.726 | 0.832 | 0.761 | 0.855 | 0.793 |
| 4 | 0.737 | 0.836 | 0.762 | 0.860 | **0.799** |

Held-out **test** rho_mean (0.818) is slightly *higher* than the final **val** (0.799) — consistent,
no signs of overfitting. eval_loss fell to 0.0087; train_loss 0.0126.

---

## CONFIG → RESULT (everything tried)

| Config | Result |
|---|---|
| deberta-v3-small, full fine-tune | ✗ crashes on MPS (disentangled attention) — switched to roberta-base |
| roberta-base full-FT, **fixed 512 padding**, bs 16 | ✗ throughput collapsed (65 s/it, memory thrash) |
| roberta-base full-FT, dynamic padding, bs 8, 1 epoch | ✓ sanity: val rho_mean **0.667** (proved pipeline + eval path) |
| roberta-base full-FT, dynamic padding, bs 8, 4 epochs | ✗ swap death-spiral (17→24 s/it, ~12 h ETA) — killed |
| **roberta-base, freeze emb + bottom 6 layers, dynamic padding, bs 8, 4 epochs** | ✅ **val 0.799 → test 0.818** (final, shipped) |

---

## FAILURE MODES HIT & FIXES APPLIED

1. **Wrong model checked in (Failure Mode #2).** `train_cuebench.py` still had
   `MODEL_NAME = "microsoft/deberta-v3-small"`, which crashes on MPS.
   → Set `MODEL_NAME = "roberta-base"`.

2. **Eval loaded no weights (Failure Mode #7).** `eval_local_model.py` only tried
   `pytorch_model.bin`, but the Trainer saves `model.safetensors` — so eval would have silently
   scored a **random-init** model and reported garbage.
   → Added a safetensors fallback that loads `model.safetensors`, prints the loaded file, and
   **fails loudly** if no weights are found. (Final load: `missing=0, unexpected=0`.)

3. **Throughput collapse → memory thrashing (NOT in the handoff — new environment issue).**
   This machine has only **17 GB RAM and was already ~9–13 GB into swap** from other apps.
   Two compounding causes:
   - **Fixed 512-token padding.** The script padded *every* sample to 512 tokens, but the token
     distribution is short-skewed (median **346**, 64% under 512, 17% under 128). Most compute
     and memory was spent on padding. → Replaced with a **`DynamicCollator`** that pads each
     batch to its own longest sequence (capped at 512). This alone took steps from **65 s/it → 2 s/it**.
   - **Full fine-tune hot working set didn't fit.** Even with dynamic padding, training all 125M
     params (grads + Adam states ≈ 2× params, all hot) plus activations exceeded available RAM,
     so it paged every step (CPU stuck at 3–18%, steps degrading 17 → 24 s/it, ~12 h ETA).
     → **Froze embeddings + bottom 6 layers** (trainable 125M → 43.7M). This halved the
     gradient + optimizer hot memory and freed frozen-layer activations, so the working set fits
     in RAM. Result: stable **~1.6 s/it**, full 4-epoch run in **~61 min**, and *better* rho.

4. **Custom Trainer eval path on transformers 5.12 (Failure Mode #6 risk).** The installed stack is
   transformers **5.12.1** (very new). Verified `MaskTrainer.prediction_step` matches the current
   `Trainer.prediction_step` signature exactly; per-epoch eval + `compute_metrics` ran cleanly with
   no API friction. No bypass needed.

Harmless warnings ignored as expected: `warmup_ratio` deprecation, `pin_memory not supported on MPS`.

---

## HONEST SUMMARY

The local model **faithfully reproduces the gpt-5.5 judge on all four axes** (rho 0.77–0.87,
mean 0.818, single-digit MAE on held-out data). There are **no "acceptable-only" or "failed" axes**
— every axis cleared the 0.7 faithfulness bar.

The two axes that were the experiment — **Delegation (0.776)** and **Discernment (0.774)**, the
"latent" ones deterministic metrics could not capture — are recovered at faithful level. Diligence
(0.870) and Description (0.851) are the strongest, as expected since they are most digest-readable.

Caveats kept honest:
- This is a faithful copy of *the judge*, not ground truth. Quality is bounded by the gpt-5.5 labels.
- Inputs are the same token-budgeted digests the judge saw; very long sessions are truncated at
  512 tokens, so the model can't see content beyond that (relevant mainly for the long tail).
- The freeze that fixed the memory problem also raised held-out rho here, but it is an environment-
  driven choice; on a machine with more RAM, full fine-tuning could be revisited (it did not obviously
  help in the 1-epoch sanity: 0.667 vs the frozen run's 0.749 at epoch 1).

**Deliverable:** `cuebench_model/` — the shippable local 4D scorer (~500 MB roberta-base).
Score any session with `python eval_local_model.py --data data/ --model cuebench_model/`.
