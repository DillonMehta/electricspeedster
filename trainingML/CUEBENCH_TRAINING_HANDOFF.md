# CueBench — Local Model Training Handoff (for Claude Code)

You are finishing an ML task that's already 90% set up. Your job: get a small
local model trained to reproduce an LLM judge's 4D scores, evaluate it on a
held-out set, and leave a short results report. Work autonomously. The human is
asleep. Don't ask questions — use the decision rules below and keep going.

---

## CONTEXT (what this is)

CueBench scores how well a human "drove" an AI coding session on **4 axes (the 4Ds)**:
- **Delegation** — how well the work was scoped/handed off
- **Description** — how well the task was specified
- **Discernment** — quality of judging/accepting the agent's output
- **Diligence** — verification/follow-through (did they test, commit cleanly)

We have ~5,781 sessions from the SWE-chat dataset, each scored 0-100 on all 4Ds
by a **gpt-5.5 LLM judge** (these scores are our ground-truth labels — treat them
as valid). The goal is to **distill that judge into a small local model** so the
4D scores can be computed on-device with **no API calls** (cost + privacy).

This is a **regression distillation**: input = a text digest of the session,
output = 4 numbers (the 4D scores, scaled 0-1). The student model can't beat the
teacher; the goal is a faithful, cheap, local copy.

---

## FILES (all in the working dir, ~/all or wherever the scripts live)

Data (already generated — do NOT regenerate unless missing):
- `swechat_rich.json` — per-session digests (prompts + signals). Input source.
- `judgments.json` — gpt-5.5 4D scores per session_id. Label source.
- `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl` — already built by
  `build_training_data.py`. Each row: `{sid, text, labels:[del,desc,disc,dil], mask:[...]}`.
  labels are 0-1 (score/100). mask=1 where the judge gave a real score, 0 where N/A.
  Coverage: delegation 5779, description 5781, discernment 4195, diligence 5561.
  All axes have real 0-1 spread (good for training).

Scripts:
- `build_training_data.py` — builds data/ from rich+judgments. Already run. Re-run
  only if data/ is missing: `python build_training_data.py --rich swechat_rich.json --judgments judgments.json --out data/`
- `train_cuebench.py` — fine-tunes the model. **This is what you're running.**
- `eval_local_model.py` — evaluates trained model on held-out test set.

---

## ENVIRONMENT (known state)

- Mac, Apple Silicon, device = **mps** (confirmed working).
- torch 2.12, transformers (recent, post-rename API), accelerate, datasets,
  sentencepiece, scipy all installed.
- The script prints `device: mps` on start. **If it ever prints `device: cpu`,
  something broke MPS — training will take hours. Investigate before letting it run.**

---

## THE IMMEDIATE TASK

Run training and evaluation to completion. Current model is **roberta-base**
(we switched FROM deberta-v3-small because DeBERTa's disentangled attention
crashes on MPS — see Failure Modes). 

### Step 1 — sanity run (1 epoch, ~5-12 min on MPS)
```
python train_cuebench.py --data data/ --out cuebench_model/ --epochs 1
```
Confirm: `device: mps`, training progress bar moves, loss decreases, and an eval
line prints with `rho_delegation/description/discernment/diligence/rho_mean`.

**Decision rule after Step 1:**
- If `rho_mean > 0.3` and at least description+diligence are climbing → proceed to Step 2 (full run).
- If `rho_mean ~ 0` or NaN → something's wrong with the loss/labels; see Failure Modes, fix, retry.
- If it crashed → see Failure Modes, apply the matching fix, retry.

### Step 2 — full run (4 epochs, ~20-50 min on MPS)
```
python train_cuebench.py --data data/ --out cuebench_model/ --epochs 4
```
Let it finish. It saves the best checkpoint by `rho_mean` to `cuebench_model/`.

### Step 3 — held-out evaluation (the proof)
```
python eval_local_model.py --data data/ --model cuebench_model/
```
This prints per-axis rho vs the judge on the test set (never seen in training).

**Interpretation (write this into the report):**
- rho ≥ 0.7 = local model faithfully reproduces the judge on that axis.
- 0.5–0.7 = usable lossy copy.
- < 0.5 = small model can't capture that axis.

EXPECTED: description + diligence high (≥0.7, they're digest-readable). delegation
+ discernment are the experiment — they're "latent" axes that deterministic metrics
could NOT capture, so even 0.5–0.7 here is a WIN (means the text model learned what
counts couldn't).

---

## KNOWN FAILURE MODES + FIXES (we hit these; apply directly)

1. **`use_mps_device` unexpected keyword** → already removed. If it reappears,
   delete that arg from `TrainingArguments`.

2. **MPS assertion: `Destination NDArray and Accumulator NDArray cannot have
   different datatype`** → this is DeBERTa-on-MPS. Already fixed by switching
   `MODEL_NAME = "roberta-base"`. If you ever see this, the model got switched
   back to a deberta variant — set it to `roberta-base` (or `distilroberta-base`).

3. **`evaluation_strategy` vs `eval_strategy`** → newer transformers uses
   `eval_strategy`. Script already uses the correct one.

4. **`warmup_ratio` deprecated** → already switched to `warmup_steps=100`. Harmless
   warning if it appears.

5. **`pin_memory not supported on MPS`** → harmless warning, ignore.

6. **MaskTrainer.prediction_step signature mismatch** (newer Trainer API): if eval
   crashes inside the custom `prediction_step`, the Trainer may call it with
   different kwargs. Fix: make the signature
   `def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):`
   and ensure it returns `(loss, predictions_tensor, (labels, mask))`. If it still
   fights the Trainer, simplest robust fallback: bypass Trainer eval — after
   `trainer.train()`, run a manual eval loop over val (like eval_local_model.py does)
   and print per-axis spearman. Don't let eval-API friction block training.

7. **Model saves as `model.safetensors`, not `pytorch_model.bin`** → `eval_local_model.py`
   tries pytorch_model.bin first. If the model won't load, load safetensors instead:
   ```python
   from safetensors.torch import load_file
   sd = load_file(f"{model_dir}/model.safetensors")
   model.load_state_dict(sd, strict=False)
   ```
   Apply this fix to eval_local_model.py if it can't find weights.

8. **OOM on MPS at batch size 16** → drop to `--bs 8` (slower, won't crash).

9. **NaN loss** → labels should be 0-1; if NaN appears, check mask isn't all-zero
   for a batch (the `.clamp(min=1)` in the loss guards this, but verify). Lower
   `--lr` to 1e-5 if unstable.

10. **DO NOT write `export OPENAI_API_KEY=...` anywhere.** (No API needed for
    training anyway — it's all local. Only eval/training of THIS model is offline.)

---

## IF roberta-base UNDERPERFORMS (rho_mean < 0.5 after full run)

Try in this order, one at a time, re-evaluating each:
1. **More epochs**: `--epochs 8` (small models often need more passes).
2. **Lower LR**: `--lr 1e-5` for stability, or `3e-5` if underfitting.
3. **Longer input**: bump `maxlen` 512 → 1024 in the DS class (captures more of
   long sessions; slower).
4. **Bigger model**: `MODEL_NAME = "roberta-large"` (~1.3GB, slower, more capable).
   Only if small one plateaus and footprint allows.
5. **Per-axis check**: if description/diligence are fine but delegation/discernment
   are the only weak ones, that's EXPECTED and acceptable — those are latent axes.
   Report them honestly rather than chasing them forever.

---

## WHAT TO LEAVE BEHIND (write these before stopping)

1. **`TRAINING_RESULTS.md`** containing:
   - final per-axis rho on the held-out test set (from Step 3)
   - which model + hyperparams produced it
   - per-axis verdict (faithful ≥0.7 / lossy 0.5-0.7 / failed <0.5)
   - the val rho_mean curve across epochs (from training logs) if available
   - any failure modes you hit and how you fixed them
   - honest summary: "the local model reproduces the judge well on [axes],
     acceptably on [axes], poorly on [axes]"

2. The trained model in `cuebench_model/` (the deliverable — this is the shippable
   local 4D scorer, ~500MB roberta-base or ~330MB if you used distilroberta).

3. If you tried multiple configs, a one-line table of config → rho_mean so the
   human can see what worked.

---

## NORTH STAR

The human spent days proving that deterministically, only 2 of the 4Ds (Diligence
via verification, Description via embedding-specificity) are cleanly measurable.
Delegation and Discernment are "latent" — they need a model that reads the text.
**This training run is the test of whether a small LOCAL model can recover those
latent axes by distilling the LLM judge.** If it gets delegation/discernment to
even 0.5-0.7 rho on held-out data, that's the whole product unlocked: a local,
no-API, full-4D scorer. Get it trained, evaluate honestly, leave the report.

Don't overfit-chase. Don't fake numbers. If an axis won't learn, say so. A real
0.55 is worth more than a fabricated 0.8. Good luck.
