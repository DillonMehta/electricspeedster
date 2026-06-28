# CueBench — Live Agent Integration Plan (for Claude Code)

Build a local agent that watches Claude Code sessions, scores each finished
session (deterministic signals + the trained model), generates a grounded trace
via BYOK, and POSTs the result to the CueBench dashboard so it updates in real time.

Work autonomously. Build it to RUN, not to be perfect — the human will tune later.
Be honest: never fabricate a signal. If you can't measure something, omit it.

---

## WHAT YOU ALREADY HAVE
- `cuebench_model/` — the trained RoBERTa 4D scorer (distills the judge, rho ~0.82
  on held-out, all 4 axes). Loads via `train_cuebench.Regressor` + tokenizer.
- `cuebench_signals.py` — consolidated deterministic signals module (provided).
  Contains: verification classifier, git signals, `build_inputs()`, `digest_text()`
  (the EXACT format the model expects), specificity (BYOK), checklist derivation, and the
  `quality` scale (confidence gate + 6 zones + EB shrinkage; replaces the old letter grade).
- `train_cuebench.py` / `eval_local_model.py` — model definition + loader reference.

## WHAT YOU MUST BUILD
A single runnable agent, `cuebench_agent.py`, that does the pipeline below, plus a
tiny `model_infer.py` helper for loading the model and scoring a digest.

---

## ⚠ THE ONE THING THAT WILL SILENTLY BREAK EVERYTHING
The model was trained on the output of `cuebench_signals.digest_text(inputs)`.
Your JSONL parser MUST produce an `inputs` dict with the SAME fields
(`prompts, n_prompts, n_tools, n_edits, verify_runs, loops, n_commits, churn,
survival_proxy, reverts`) so `digest_text()` yields the same string shape the model
saw in training. Use `cuebench_signals.build_inputs(...)` to assemble it — do not
hand-roll the digest. If the digest format drifts, scores are garbage and nothing
will error. Verify by printing a digest and eyeballing it against a training row in
`data/train.jsonl`.

---

## PIPELINE (build in this order)

### Step 1 — Find & read finished Claude Code sessions
Claude Code writes a JSONL transcript per session. Locate the transcripts dir
(commonly `~/.claude/projects/<project-hash>/<session-uuid>.jsonl` — INSPECT the
real path on this machine first; `find ~/.claude -name "*.jsonl" | head` and open
one to learn its actual line schema before parsing).

A session is "finished" when its JSONL file stops growing (no new lines for ~60s).
Implement a watcher: poll the dir every 15-30s, track file sizes/mtimes, and when a
file has been stable for 60s and hasn't been scored yet, process it. Keep a small
`scored_sessions.json` set of session IDs already POSTed so you never double-score.

### Step 2 — Parse the JSONL into the signal inputs
From the session's JSONL lines, extract:
- `prompts`: the human/user messages (the actual prompt text the operator typed).
- `tool_cmds`: list of bash commands run (tool_use entries with a command/input).
  Used for `verify_runs` via the classifier.
- `n_edits`: count of file-edit/write tool calls.
- `loops`: heuristic — count near-identical consecutive tool calls / repeated
  failing actions. Simple version: consecutive tool calls with the same target +
  similar args. Start crude; tune later.
- model string, token counts (input/output if present in the JSONL), timestamps
  (first→last for duration), and the repo/working dir (to run git on).

INSPECT a real JSONL to map these — the exact keys depend on Claude Code's format.
Build the parser defensively: tolerate missing fields, log what you couldn't find.

### Step 3 — Git signals
Determine the repo the session worked in (from cwd in the JSONL, or pass it in).
Call `cuebench_signals.git_signals(repo_dir, since_ref=<session-start-sha-if-known>)`.
If you can't determine a start SHA, the function falls back to recent commits — fine
for v1. This fills `n_commits, churn, reverts, survival_proxy`.

### Step 4 — Assemble inputs + digest + SCORE WITH THE MODEL
```python
from cuebench_signals import build_inputs, digest_text, derive_checklist, quality, specificity
inputs = build_inputs(prompts, tool_cmds, n_edits, loops, git)
digest = digest_text(inputs)
vectors = model_infer.score(digest)   # -> {"delegation":int,...} 0-100 each
score = round(sum(vectors.values())/4)
```
`model_infer.score(digest)`: load `cuebench_model/` once at startup (RoBERTa via
`Regressor`, safetensors load — see eval_local_model.py / Failure Mode notes),
tokenize digest (maxlen 512), forward, multiply the 4 sigmoid outputs ×100, round.
Keep the model loaded in memory; score many sessions without reloading.

### Step 5 — Deterministic cross-checks + checklist
- `checklist = derive_checklist(inputs, vectors)` — grounded pass/fail from real signals.
- Optionally compute `specificity(prompts, openai_client)` as a Description
  cross-check (BYOK key). Store it but the model's Description score is primary.
- Build `breakdown` array in EXACT order: delegation, description, discernment, diligence.

### Step 6 — BYOK trace generation (grounded, not fabricated)
The CueBench `trace` (timeline moments) and richer `detail` text are GENERATED, not
measured — so use the org's BYOK LLM key to produce them FROM THE REAL TRANSCRIPT.
This is legitimate because it's grounded summarization of what actually happened,
not invention.

- Read BYOK key + model from env: `CUEBENCH_BYOK_KEY`, `CUEBENCH_BYOK_MODEL`
  (default e.g. "claude-opus-4" or "gpt-4o" — provider inferred from key/model).
- Prompt the BYOK model with: the parsed prompts + tool sequence + timestamps +
  the computed signals, and ask for 4-8 trace items in CueBench shape
  `{t:"MM:SS", sig:"good|warn|info", label:"≤6 words", detail:"one grounded sentence"}`.
  Instruct it to reference ACTUAL files/tools/actions from the transcript and to
  mark loops/verification/corrections it can see. Require strict JSON output; parse
  defensively; if it fails or no BYOK key is set, fall back to a minimal trace built
  from deterministic signals (e.g. "verification run", "loop detected") so the POST
  still succeeds.
- Do NOT invent token counts, baselines, or savings. Send real token counts from the
  JSONL; OMIT `baseline`, `timeBaseline`, `savings` if not measured (they're optional).

### Step 7 — Assemble the CueBench payload & POST
Map everything to the POST /api/session schema (see the CueBench Integration Brief).
Required: employeeId, sessionId, task, model, duration, score, quality, vectors, trace,
cost(input/output/tool/total/time — comma-formatted strings), breakdown, checklist.
- `sessionId`: "S-" + session-uuid first 8 chars uppercased.
- `task`: 4-6 word title — derive from the first prompt (BYOK can summarize it; or
  take a deterministic truncation of the first prompt). Grounded, no filler.
- `quality` (replaces the old `grade`): `cuebench_signals.quality(raw_composite, inputs)` — a
  confidence gate first ("insufficient_signal" for thin sessions, NOT zoned) else one of 6 quality
  zones from the EB-shrunk composite. See cuebench_signals §7 and AGENT_README "Quality scale".
- `cost.tool`: if you can't isolate tool tokens, use ~15% of total.
- POST to `https://app.cuebench.dev/api/session` with header
  `x-api-key: <CUEBENCH_API_KEY env>`. On success the dashboard updates via WebSocket.

### Step 8 — Employee identity
Read `CUEBENCH_EMPLOYEE_ID` env (default "e1" for this machine). If onboarding a new
operator, the human will POST /api/employees separately — you just use the ID.

---

## SECRETS — read from env ONLY, never hardcode
- `CUEBENCH_API_KEY`   — dashboard write key (the human will set + ROTATE it; it leaked in chat).
- `CUEBENCH_EMPLOYEE_ID` — default "e1".
- `CUEBENCH_BYOK_KEY` / `CUEBENCH_BYOK_MODEL` — for trace generation + specificity.
Do NOT write `export OPENAI_API_KEY=...` lines into any script (it has wrecked the
human's real key before). Read keys with os.environ.get; if a key is missing,
degrade gracefully (skip BYOK trace/specificity, still POST the core score).

---

## RUN SHAPE
`python cuebench_agent.py` should: load model once, start the watcher loop, and for
each newly-finished session run Steps 2-7 and POST. Print a line per scored session
(sid, task, score, quality, POST status). Add `--once <path-to-jsonl>` mode to score a
single transcript for testing without waiting on the watcher.

## TEST IT END-TO-END BEFORE DECLARING DONE
1. `--once` on a real recent JSONL: confirm a digest prints that looks like a
   training row, the model returns 4 plausible scores, checklist/trace build, and the
   assembled payload validates against the schema.
2. Dry-run POST (print payload) first; then real POST with the key set; then
   `GET https://app.cuebench.dev/api/data` and confirm `data.session` is your session
   and the employee score moved.
3. Leave a short `AGENT_README.md`: how to set the 4 env vars, how to run, how to
   point it at a different repo, and any JSONL-format assumptions you made (so the
   human can fix them if Claude Code's format differs).

## KNOWN GOTCHAS
- Model loads as `model.safetensors` (not pytorch_model.bin) — use safetensors load.
- Keep the model on CPU for the agent (no need for MPS at inference; one session
  scores in <1s on CPU). Simpler + no MPS quirks.
- The JSONL schema is the biggest unknown — INSPECT a real file first, don't assume.
- digest_text format is sacred (see the big warning above).
- Be honest in the checklist/trace: grounded in real signals/transcript only.

## NORTH STAR
Something running: a finished Claude Code session shows up on the dashboard within a
minute or two, with a real model score, real deterministic checklist, and a BYOK trace
grounded in the actual transcript. Tune accuracy later. Don't fake fields to look
complete — omit what isn't measured.
