# CueBench Live Agent

Watches finished Claude Code sessions, scores each on the 4Ds with the **local trained
model** + **deterministic git/verification signals**, generates a **grounded BYOK trace**
from the real transcript, and **POSTs** the result to the CueBench dashboard.

Built to **run**, not to be perfect — tune the heuristics later. Everything is grounded in
real signals; nothing is fabricated. Fields that can't be measured are omitted.

---

## Files

| File | Role |
|---|---|
| `cuebench_agent.py` | The agent: watcher loop, JSONL parser, payload assembly, POST. |
| `model_infer.py` | Loads `cuebench_model/` once (CPU) and scores a digest → 4 axis scores. |
| `cuebench_signals.py` | (provided) `build_inputs` / `digest_text` / `git_signals` / `derive_checklist` / `quality` (confidence gate + 6-zone scale) / `specificity`. |
| `train_cuebench.py` | (provided) model definition (`Regressor`, `AXES`) — imported by `model_infer`. |
| `cuebench_model/` | (provided) the trained RoBERTa 4D scorer (`model.safetensors`). |
| `scored_sessions.json` | written at runtime — set of already-POSTed sessions (dedup). |

---

## Setup

```bash
# Required to POST. The human sets + ROTATES this (it leaked in chat).
export CUEBENCH_API_KEY="..."

# Optional (sensible defaults shown)
export CUEBENCH_EMPLOYEE_ID="e1"          # which operator this machine is
export CUEBENCH_BYOK_KEY="..."            # enables grounded LLM trace + specificity
export CUEBENCH_BYOK_MODEL="claude-opus-4-8"   # provider inferred from model/key
```

**Never hardcode keys** — the agent reads them from the environment only and degrades
gracefully when they're missing (skips BYOK trace/specificity, still POSTs the core score).

### BYOK provider

`CUEBENCH_BYOK_MODEL` decides the provider:
- model contains `claude` **or** key starts with `sk-ant` → **Anthropic** (uses the `anthropic` SDK).
- otherwise → **OpenAI** (uses the `openai` SDK, e.g. `gpt-4o`).

> The `openai` SDK is already installed. The `anthropic` SDK is **not** — for the Claude
> BYOK path run `pip install anthropic`. Without it (or without any BYOK key) the agent
> falls back to a deterministic trace built from the measured signals, and the POST still
> succeeds. `specificity` (a Description cross-check) needs an **OpenAI** key + embeddings;
> it's skipped otherwise. It's stored as `specificity`, never overrides the model's score.

---

## Running

```bash
# Watcher (the normal mode): score every newly-finished session and POST it.
# Requires CUEBENCH_API_KEY (exits if unset — it would otherwise score but never POST).
python cuebench_agent.py

# Score one transcript without POSTing (testing) — prints the full payload:
python cuebench_agent.py --once ~/.claude/projects/<proj>/<session>.jsonl --dry-run

# Score one transcript and POST it (needs CUEBENCH_API_KEY):
python cuebench_agent.py --once ~/.claude/projects/<proj>/<session>.jsonl
```

The watcher polls `~/.claude/projects/**/*.jsonl` every `CUEBENCH_POLL_INTERVAL`s (default 20);
a transcript is "finished" once its file size hasn't changed for `CUEBENCH_STABLE_SECONDS`
(default 60). Each finished, not-yet-scored session runs the full pipeline and POSTs. It
prints one line per session: `sid · title · score · quality (zone or "Insufficient signal") · gen · POST status`.

> First start loads the model once (a few seconds; can be slower on this RAM-constrained Mac
> under memory pressure). After that, each session scores in well under a second.

### Pointing it at a different repo

The agent reads the repo from each session's `cwd` field in the JSONL (the directory you ran
Claude Code in) and runs `git` there automatically — no flag needed. To watch transcripts
stored elsewhere, set `CUEBENCH_PROJECTS_DIR`.

### All env knobs

| Var | Default | Meaning |
|---|---|---|
| `CUEBENCH_API_KEY` | — | **Required for POST.** `x-api-key` header. |
| `CUEBENCH_EMPLOYEE_ID` | `e1` | `employeeId` in the payload. |
| `CUEBENCH_BYOK_KEY` / `CUEBENCH_BYOK_MODEL` | — / `claude-opus-4-8` | BYOK trace + task title + specificity. |
| `CUEBENCH_API_URL` | `https://app.cuebench.dev/api/session` | POST target. |
| `CUEBENCH_DATA_URL` | `https://app.cuebench.dev/api/data` | (for your verification GET) |
| `CUEBENCH_PROJECTS_DIR` | `~/.claude/projects` | where transcripts live. |
| `CUEBENCH_MODEL_DIR` | `cuebench_model` | trained model dir. |
| `CUEBENCH_SCORED_FILE` | `scored_sessions.json` | dedup store. |
| `CUEBENCH_POLL_INTERVAL` | `20` | seconds between dir scans. |
| `CUEBENCH_STABLE_SECONDS` | `60` | no-growth window → "finished". |

---

## Verify end-to-end (recommended first run)

1. **Dry-run on a recent real transcript** and eyeball the printed `_meta.digest_preview` — it
   must look like a training row (`PROMPTS=… TOOLS=… …\nPROMPTS: [P1] …`). Confirm the 4 vectors
   are plausible and the checklist/trace built:
   ```bash
   python cuebench_agent.py --once <recent>.jsonl --dry-run
   ```
2. **Real POST** with the key set, then confirm on the dashboard:
   ```bash
   export CUEBENCH_API_KEY="..."
   python cuebench_agent.py --once <recent>.jsonl
   curl -s -H "x-api-key: $CUEBENCH_API_KEY" "$CUEBENCH_DATA_URL" | python -m json.tool | grep -A3 <your-sid>
   ```
   The dashboard updates over WebSocket on success.
3. **Start the watcher** and let new sessions flow in automatically.

> **Note on this build:** I validated the full pipeline with `--once --dry-run` on several real
> transcripts and the watcher mechanics on a scratch dir. I did **not** perform a live POST —
> `CUEBENCH_API_KEY` is not in the environment (you're rotating the leaked key). `GET /api/data`
> returns HTTP 200, so the endpoint is reachable. Do step 2 once you've set the key.

---

## What gets sent (payload shape)

No CueBench Integration Brief was present in the repo, so the payload shape below was assembled
from the plan's Step 7 field list. If the live schema differs, adjust `build_payload()` in
`cuebench_agent.py` — the values are all grounded; only the JSON shape may need tweaking.

```jsonc
{
  "employeeId": "e1",
  "sessionId": "S-1112BF4C",          // "S-" + first 8 of session uuid, uppercased
  "task": "Add debounce to search",   // BYOK 4-6 word title, else truncated first prompt
  "model": "claude-opus-4-8",         // from the transcript
  "duration": "20m 41s",
  "score": 42,                        // round(mean of the 4 vectors)  (raw composite)
  // Headline verdict (replaces the old letter grade). cuebench_signals.quality():
  // a confidence gate FIRST — thin sessions get state "insufficient_signal" and are
  // NOT zoned — else one of 6 quality zones from the EB-shrunk composite.
  "quality": {
    "state": "assessed",              // or "insufficient_signal" (gated, not a zone)
    "zone": 4, "key": "inconsistent", "label": "Inconsistent",
    "definition": "Quality swings — uneven work, targeted practice needed.",
    "urgent": false,                  // true only for zones 5 (Needs attention) & 6 (Critical)
    "support": 137,                   // n_effective = prompts + tool calls
    "eb_score": 47, "raw_score": 42   // eb_score is the zoning basis (== raw for busy sessions)
  },
  "vectors": { "delegation": 64, "description": 53, "discernment": 33, "diligence": 19 },
  "trace": [ { "t":"00:00","sig":"info","label":"…","detail":"…" }, … ],  // BYOK or deterministic
  "cost": { "input":"774,477","output":"15,461","tool":"118,491","total":"789,938","time":"20m 41s" },
  "breakdown": [ { "axis":"delegation","score":64,"checklist":[…] }, … ],  // EXACT axis order
  "checklist": { "delegation":[…], "description":[…], "discernment":[…], "diligence":[…] },
  "specificity": 71,                  // optional, only with an OpenAI BYOK key
  "_meta": { "digest_preview":"…", "repo":"…", "git_branch":"…", "trace_source":"byok|deterministic" }
}
```

Token counts are **real** (summed from the transcript's `usage`, input includes cache tokens).
`cost.tool` is ~15% of total (the plan's fallback — tool tokens can't be isolated from the JSONL).
`baseline` / `timeBaseline` / `savings` are **omitted** (not measured — don't fabricate them).
`_meta` is local diagnostics; harmless if the API ignores unknown fields — drop it if it rejects them.

---

## Quality scale (the `quality` field — replaces letter grades)

The headline a reviewer sees is a **quality zone**, not a letter. Two stages, in `cuebench_signals.quality()`:

1. **Confidence gate (first).** `support` = `n_effective` = operator prompts + tool calls. Below
   `THIN_SESSION_MIN_SUPPORT` there isn't enough behaviour to assess the four axes reliably, so the
   session gets the **separate** state `insufficient_signal` ("too short/thin to assess reliably") and
   is **not** placed on the scale. Insufficient signal is **not** one of the 6 zones.
2. **Zoning.** Substantive sessions are placed by their **EB-shrunk** composite. Empirical-Bayes
   shrinkage pulls a raw composite toward the population mean in proportion to how thin the session is
   (`eb = (support·raw + K·prior)/(support+K)`), so a few lucky/unlucky actions can't fling a borderline
   session into the top or bottom zone. Busy sessions keep their raw score.

| # | Zone | Tier | Definition |
|---|---|---|---|
| 1 | Dialed in | — | Strong, deliberate driving across the axes. |
| 2 | Solid | — | Effective driving — a few areas to sharpen. |
| 3 | Developing | — | Fundamentals present, with clear headroom. |
| 4 | Inconsistent | — | Quality swings — uneven work, targeted practice needed. |
| 5 | Needs attention | **URGENT** | Driving is limiting outcomes — act now. |
| 6 | Critical | **URGENT** | Core practices breaking down — address before the next session. |

The zone is the **headline**; insights/trace are the **detail** and are generated to stay coherent with
it (the verdict is fed to the generator — no "Critical" zone with mild, congratulatory insights).

**Boundaries are NOT academic cutoffs.** All thresholds are named constants in `cuebench_signals.py`
(`THIN_SESSION_MIN_SUPPORT`, `EB_PRIOR_MEAN`, `EB_PRIOR_STRENGTH`, `ZONE_*_MAX`), read off the EB-shrunk
distribution of the real labelled corpus so each zone's definition fits the sessions landing in it and the
urgent tier (5+6) stays a small minority (~9% of substantive sessions in v1). They are **v1 estimates** —
re-run `python cuebench_calibrate_zones.py` to refresh them as more sessions accrue.

---

## JSONL format assumptions (verified against a real `~/.claude` transcript)

If Claude Code's format changes, fix `parse_transcript()` in `cuebench_agent.py`:

- One JSON object per line. Lines have a top-level `type`.
- **Operator prompts**: `type=="user"` lines **without** a `toolUseResult` key, whose
  `message.role=="user"`. Content is a string or a list of `{type:"text"}` blocks. We strip
  `<system-reminder>` / local-command wrappers; tool-result user messages (which carry
  `toolUseResult` + `tool_result` blocks) are excluded.
- **Bash commands**: `type=="assistant"` → `message.content[]` blocks with `type=="tool_use"`,
  `name=="Bash"` → `input.command`. These drive `n_tools` and `verify_runs`.
- **Edits**: `tool_use` blocks named `Edit` / `Write` / `MultiEdit` / `NotebookEdit`
  (confirmed `Edit` and `Write` dominate in real transcripts). → `n_edits`.
- **Model / tokens**: `message.model`, `message.usage` (`input_tokens` + `cache_read_input_tokens`
  + `cache_creation_input_tokens` for input; `output_tokens` for output).
- **Timestamps**: ISO-8601 `Z` on most lines → duration. **cwd / gitBranch**: read from any line.
- **session uuid**: `sessionId` field, else the filename stem. → `sessionId` = `S-<UUID8>`.

---

## Known limitations / tuning knobs (intentional v1 choices)

- **`loops`** is crude: count of consecutive near-identical tool calls (same tool + same
  target). Tune `detect_loops()` for repeated-failing-action detection.
- **`n_tools` = number of Bash commands** (per the provided `build_inputs`). Sessions that
  edit files via the Edit/Write tools rather than bash will show `TOOLS` lower than total tool
  calls — this matches how the model was trained (digest format is sacred; don't change it).
- **`survival_proxy` is `None` when the session made no commits** → digest shows `SURVIVAL=None`.
  That's honest (read-only sessions have no surviving lines to measure), not a bug.
- **git signals** come from `cuebench_signals.git_signals` with a best-effort session-start ref
  (`git rev-list --before=<first_ts>`). If the start ref can't be found it falls back to the
  last ~20 commits. Churn can read 0 on some histories — a known v1 rough edge in the provided
  module, not the agent.
- **Dedup**: `scored_sessions.json` stores POSTed sessions (and un-scoreable ones) by path/sid so
  they're never re-POSTed. Transient POST failures are **not** marked — they retry on a later poll.
  Delete the file to force a full rescan.
- **Subagent work** (Task/Agent tool spawns) lives in separate transcripts; this agent scores
  each transcript independently and does not stitch parent/child sessions together.
