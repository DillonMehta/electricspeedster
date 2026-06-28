#!/usr/bin/env python3
"""
cuebench_gen.py — neutral, privacy-preserving BYOK generation (title / insights / trace)
========================================================================================
ONE generation module. ONE provider switch (anthropic | openai). ONE key/provider
setting governs ALL THREE generations. Generation is OFF when no key is set, and a
generation failure NEVER blocks scoring or the POST.

PRIVACY (non-negotiable)
------------------------
Raw prompts, raw code, file paths, and verbatim commands must NEVER leave the device.
The generator is fed ONLY derived numbers — the same deterministic signals the score is
built from (counts, rates, scores) — and NEVER raw session text. That makes neutrality a
STRUCTURAL guarantee: the model cannot echo content it was never given. The neutrality
rule is also baked into the instructions as defense-in-depth.

Two generations are content-aware opt-ins (everything else stays numbers-only; the session
TITLE is no longer generated here — it is produced locally by cuebench_classify):
  • trace()         — fed a PRE-EXTRACTED, TIMESTAMPED EVENT MENU from the whole session
                      timeline (cuebench_agent.build_session_events): opening/steering prompts,
                      tool loops, the context build, edits, verify runs, model, close. The model
                      labels the notable events and references each by its integer id; the TIME on
                      every entry is computed by the CALLER and stamped from that id, so the model
                      never sees or authors a timestamp — and can only cite an event we extracted.
  • insights()      — numbers-only BY DEFAULT, but OPT-IN "prompt-informed" mode
                      (CUEBENCH_INSIGHTS_PROMPTS) additionally feeds the operator's own
                      PROMPTS so the coaching can reference what was actually asked / how it
                      was scoped. Off -> numbers-only and structurally neutral as before.
All three send raw content ONLY to the user's own BYOK endpoint (exactly like specificity);
only the generated text (a grounded timeline; grounded coaching) reaches the payload — the raw
inputs never do, and trace times are computed locally, not by the model. trace and prompt-
informed insights are additionally double-gated (off unless their flag is set). _metrics_block
ALWAYS stays numbers-only: do NOT add raw prompts/paths/commands to it — prompt-informed
insights appends prompts to the user message separately, on the opt-in path only.

Config (env, per machine — never hardcode a key, never ship a default key)
--------------------------------------------------------------------------
  CUEBENCH_BYOK_KEY        the API key (absent -> generation OFF)
  CUEBENCH_BYOK_PROVIDER   "anthropic" | "openai"  (optional; else inferred from key/model)
  CUEBENCH_BYOK_MODEL      model id (optional; cheap per-provider default otherwise)
  CUEBENCH_TRACE           "1"/"true"/"on" to enable the optional trace (OFF by default)
  CUEBENCH_INSIGHTS_PROMPTS "1"/"true"/"on" -> prompt-informed insights: also feed the
                           operator's own prompts so coaching can reference what was asked
                           (content-aware opt-in; OFF by default -> numbers-only insights)

Cheap defaults: Anthropic -> claude-haiku-4-5 ; OpenAI -> gpt-4o-mini.
Per-provider request shapes differ (Anthropic: system kwarg + max_tokens; OpenAI: system
message + max_completion_tokens) — both are handled below.
"""
from __future__ import annotations
import json, os, re, sys

AXES = ("delegation", "description", "discernment", "diligence")

# Cheap models per provider. Anthropic Haiku-tier is the cheapest Claude model; Haiku does
# NOT accept thinking/effort params, so we never send them.
_DEFAULT_MODEL = {"anthropic": "claude-haiku-4-5", "openai": "gpt-4o-mini"}

# Baked-in neutrality contract — the safeguard, applied to every generation.
_NEUTRALITY = (
    "You are a coding-process analyst for CueBench. You are given ONLY anonymous, derived "
    "metrics about ONE agentic coding session — never the source code, the prompts, file "
    "names, or commands. Describe DRIVING BEHAVIOR only: how the work was delegated, "
    "specified, reviewed, and followed through. NEVER describe the subject matter. Do NOT "
    "invent or guess specifics of any kind — no client names, product names, file paths, "
    "programming languages, frameworks, domains, or proprietary terms. If a fact is not in "
    "the metrics you were given, do not state it. "
    "GOOD: 'The task description lacked explicit acceptance criteria.' "
    "BAD: 'The billing refactor was vague.'"
)

# Prompt-informed insights (opt-in) swaps the numbers-only neutrality preamble for one that
# LETS the model read the operator's own prompts, so coaching can name what was asked and how
# it was scoped. Same content-aware class as content_title/trace: the prompts go ONLY to the
# user's own BYOK endpoint, but the generated insight TEXT — which may now reference the task —
# does enter the payload. OFF by default; numbers-only neutrality applies otherwise.
_INSIGHTS_CONTENT_PREAMBLE = (
    "You are a coding-process analyst for CueBench. You are given derived metrics about ONE "
    "agentic coding session AND the operator's own prompts from that session. Use the prompts "
    "to make coaching CONCRETE: you MAY reference what the operator asked for and how they "
    "scoped, sequenced, and phrased it. Keep the focus on DRIVING BEHAVIOUR — how the work was "
    "delegated, specified, reviewed, and followed through — not on the subject matter for its "
    "own sake. Use ONLY facts present in the prompts or metrics; do NOT invent specifics, and "
    "do NOT quote long verbatim passages — paraphrase briefly and tie behaviour to what was "
    "actually requested."
)


# Keeps every generation coherent with the headline verdict the dashboard shows: the
# zone is the headline, the generated text is the detail, and they must not contradict
# (no "Critical" zone paired with mild, congratulatory insights).
def _coherence_clause(quality: dict | None) -> str:
    if not quality:
        return ""
    if quality.get("state") == "insufficient_signal":
        return (" NOTE: this session was too short/thin to assess reliably (Insufficient "
                "signal). Keep any output brief and tentative; do not over-claim strengths "
                "or weaknesses from sparse activity.")
    label = quality.get("label")
    definition = quality.get("definition") or ""
    if not label:
        return ""
    base = (f" The session's overall verdict is '{label}'"
            + (f" — {definition}" if definition else "") +
            ". Your output MUST be coherent with this verdict and match its severity")
    if quality.get("urgent"):
        return (base + ": lead with direct, specific improvements and keep strengths "
                "minimal and honest. Do not soften an urgent verdict with congratulatory "
                "framing.")
    return base + "; do not contradict it (e.g. no glowing praise for a low verdict, no harsh framing for a strong one)."


def _infer_provider(key: str | None, model: str | None) -> str:
    """Infer provider from the key/model when CUEBENCH_BYOK_PROVIDER is unset."""
    k, m = (key or "").lower(), (model or "").lower()
    if k.startswith("sk-ant") or "claude" in m:
        return "anthropic"
    return "openai"


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def _t_to_secs(t: str) -> int:
    """'MM:SS' or 'H:MM:SS' -> elapsed seconds, for sorting the trace chronologically.
    Tolerant: unparseable timestamps sort first (0)."""
    try:
        parts = [int(x) for x in str(t).split(":")]
    except Exception:
        return 0
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] if parts else 0


def _extract_json(text: str):
    """Pull the first JSON object/array out of a model response; tolerant of code fences."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = text.find(open_c), text.rfind(close_c)
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                pass
    return None


class Generator:
    """Provider-switched generator with graceful degradation.

    `enabled` is False when no key is set: title falls back to the sessionId (caller's job),
    insights/trace are omitted. Every generation method returns None on any failure so a
    caller can degrade without blocking scoring or the POST.
    """

    def __init__(self, provider: str | None = None, key: str | None = None,
                 model: str | None = None, trace_enabled: bool | None = None,
                 insights_prompts_enabled: bool | None = None):
        self.key = (key if key is not None else os.environ.get("CUEBENCH_BYOK_KEY")) or None
        env_model = os.environ.get("CUEBENCH_BYOK_MODEL")
        self.provider = (provider or os.environ.get("CUEBENCH_BYOK_PROVIDER")
                         or _infer_provider(self.key, env_model))
        self.model = model or env_model or _DEFAULT_MODEL.get(self.provider, "")
        self.trace_enabled = (_truthy(os.environ.get("CUEBENCH_TRACE"))
                              if trace_enabled is None else bool(trace_enabled))
        # NOTE: specificity is no longer a BYOK feature — it is computed LOCALLY via the
        # on-device encoder (cuebench_classify). No OpenAI embeddings, no raw text off-device.
        # prompt-informed insights = feed the operator's own prompts to insights() so coaching
        # can reference what was asked. Content-aware opt-in (like trace); OFF by default.
        self.insights_prompts_enabled = (_truthy(os.environ.get("CUEBENCH_INSIGHTS_PROMPTS"))
                                         if insights_prompts_enabled is None else bool(insights_prompts_enabled))
        self.calls = 0          # count of BYOK API calls actually issued (dry-run proof)

    @property
    def enabled(self) -> bool:
        return bool(self.key)

    def status(self) -> str:
        if not self.enabled:
            return "off (no CUEBENCH_BYOK_KEY)"
        return (f"{self.provider}:{self.model} trace={'on' if self.trace_enabled else 'off'} "
                f"insight-prompts={'on' if self.insights_prompts_enabled else 'off'}")

    # --- the single provider switch -------------------------------------------------
    def _complete(self, system: str, user: str, max_tokens: int = 600) -> str | None:
        """One completion. Returns raw text, or None on any failure (degrade gracefully).

        Per-provider request shapes differ — Anthropic takes `system` as a kwarg and uses
        `max_tokens`; OpenAI takes a system message and uses `max_completion_tokens`.
        """
        if not self.enabled:
            return None
        self.calls += 1
        try:
            if self.provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=self.key)
                resp = client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,                 # Anthropic param (NOT max_completion_tokens)
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            else:
                import openai
                client = openai.OpenAI(api_key=self.key)
                resp = client.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=max_tokens,      # OpenAI param
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                )
                return resp.choices[0].message.content
        except Exception as e:
            print(f"  [gen] {self.provider} call failed ({e!r}); degrading gracefully",
                  file=sys.stderr)
            return None

    # --- the BYOK generations (numbers in, neutral text out) ------------------------
    # NOTE: ALL session-TITLE generation was REMOVED from BYOK (both the neutral metrics title
    # and the content_title from the first prompt). The session title is now produced locally
    # by cuebench_classify (the operator's own first prompt, normalized) — deterministic, free,
    # no prompt sent off-device for naming. BYOK now powers only insights / trace / specificity.

    def insights(self, metrics: dict, vectors: dict, quality: dict | None = None,
                 prompts: list[str] | None = None) -> dict | None:
        """{'strengths':[{axis,point}...], 'improvements':[{axis,point}...]} or None.

        2-3 of each; every item tied to one of the four axes. Each `point` must be SPECIFIC
        (cite the metric value that justifies it) and, for improvements, ACTIONABLE (end with
        a concrete behaviour to do next session) — generic coaching like "improve diligence"
        is explicitly disallowed. The tone is kept coherent with the headline quality zone
        (see _coherence_clause): a 'Critical'/'Needs attention' verdict must not read as mild
        encouragement.

        PROMPT-INFORMED (opt-in): when CUEBENCH_INSIGHTS_PROMPTS is set AND `prompts` is
        passed, the operator's own prompts are appended to the user message so coaching can
        reference what was asked / how it was scoped. The prompts go ONLY to the user's own
        BYOK endpoint, but the resulting insight TEXT may then reference the task and DOES
        enter the payload (same content-aware tradeoff as content_title/trace). OFF by default
        -> numbers-only and structurally neutral exactly as before.
        """
        use_prompts = bool(self.enabled and self.insights_prompts_enabled and prompts)
        preamble = _INSIGHTS_CONTENT_PREAMBLE if use_prompts else _NEUTRALITY
        # Rule 4 toggles with the mode: numbers-only forbids subject matter outright; the
        # prompt-informed preamble already permits referencing the task, so allow it but keep
        # each point anchored to behaviour (not a content summary).
        rule_4 = ("4. You MAY reference what was asked (from the prompts) to make a point "
                  "concrete, but keep each point about DRIVING BEHAVIOUR and never quote long "
                  "verbatim text.\n") if use_prompts else \
                 "4. Behaviour only — never the subject matter.\n"
        system = preamble + (
            " Produce SPECIFIC, ACTIONABLE per-session coaching as STRICT JSON: "
            '{"strengths":[{"axis":"<one of delegation|description|discernment|diligence>",'
            '"point":"<text>"}],"improvements":[{"axis":"...","point":"..."}]}. '
            "Give 2-3 strengths and 2-3 improvements, each point AT MOST 2 sentences. "
            "Output ONLY JSON.\n"
            "RULES FOR EVERY point — this is what makes coaching useful, not filler:\n"
            "1. CITE THE NUMBER. Ground each point in the specific metric value(s) that "
            "justify it (e.g. 'verified 14 times across 21 edits', '0 tests over 88 tool "
            "calls', '22 duplicate re-runs — 16% of tool calls', 'churn of 5,200 written "
            "lines'). A point that names no number is too vague — do not emit it.\n"
            "2. IMPROVEMENTS LEAD WITH THE FIX. Start each improvement with the concrete "
            "action to take next session, phrased as a directive (e.g. 'Run the test suite "
            "after each edit, not just at the end', 'Open one prompt per task with explicit "
            "acceptance criteria', 'Commit in smaller increments so a bad change is cheap to "
            "roll back'). The FIX is the point — state it first, then justify it in at most "
            "one short clause with the number; do NOT dwell on the problem. The reader cares "
            "about what to do, not a description of what went wrong. BANNED: vague directives "
            "('optimize', 'be more careful', 'improve diligence', 'engage more') and busywork "
            "that isn't real coaching ('make at least one edit'). If a low number simply "
            "reflects the session type (e.g. a read/plan session legitimately has few edits), "
            "it is NOT a weakness — do not manufacture an improvement for it.\n"
            "3. Tie each item to the axis whose score/signals most support it.\n"
            + rule_4 +
            "SIGNAL GLOSSARY (what the numbers mean): verify_runs = test/typecheck/lint/"
            "build commands; loops = exact-duplicate tool re-runs (retrying the same thing); "
            "churn = total written lines (large => big, hard-to-review changes); n_commits / "
            "reverts = commit discipline; survival_proxy = fraction of commits not reverted; "
            "n_prompts = operator steering turns; n_edits = file edits; n_tools = all tool "
            "calls. Derive ratios where they sharpen a point (verify-per-edit, loop rate, "
            "churn-per-edit).\n"
            "EXAMPLE good improvement (fix first, problem in one clause): 'Run a typecheck or "
            "the test suite after each substantive edit so regressions surface in-session — "
            "this run made 46 edits and 5,200 lines of churn with 0 verification runs.' "
            "EXAMPLE bad improvement (problem-first, vague fix — never do this): 'There was a "
            "lot of churn and no testing this session; consider being more careful and "
            "testing more.'"
        ) + _coherence_clause(quality)
        user = self._metrics_block(metrics, vectors, quality)
        if use_prompts:                              # opt-in path: append prompts to the USER msg
            user += "\n\n" + self._prompts_block(prompts)   # (NOT _metrics_block — that stays numbers-only)
        raw = self._complete(system, user, max_tokens=900)
        data = _extract_json(raw) if raw else None
        if not isinstance(data, dict):
            return None
        out = {}
        for bucket in ("strengths", "improvements"):
            items = data.get(bucket)
            clean = []
            if isinstance(items, list):
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    axis = str(it.get("axis", "")).strip().lower()
                    point = str(it.get("point", "")).strip()
                    if axis in AXES and point:
                        # 400-char safety net: the "<=2 sentences" rule keeps points short;
                        # this only bounds pathological output (prompt-informed points cite
                        # the phrasing AND a directive, so they run longer than numbers-only).
                        clean.append({"axis": axis, "point": point[:400]})
            out[bucket] = clean[:3]
        if not out.get("strengths") and not out.get("improvements"):
            return None
        return out

    def trace(self, session: dict | None) -> list | None:
        """FACTUAL session trace built from a PRE-EXTRACTED, TIMESTAMPED EVENT MENU.

        `session` (cuebench_agent.build_session_events) is {"duration","totals","events"} where
        each event is a real moment from the transcript — the opening prompt, a later steering
        prompt, a tool loop (target + count + turn range), the context build, a file's edits, a
        verify run, the model, or the session close — carrying an integer `id` and a code-computed
        `t`. The model picks the 4-8 most notable events and writes sig/label/detail, referencing
        each by its `id`; we stamp t = event[id].t (GROUND TRUTH from the transcript, never
        model-authored) and DROP any entry whose id we didn't extract — so the trace can never
        cite a moment that didn't happen. CAMERA, not a judge: no scoring/competency vocabulary.
        Raw content (prompt text, file basenames, commands) goes ONLY to the user's own BYOK
        endpoint; only the short labels reach the payload. Double-gated: off unless CUEBENCH_TRACE
        is set AND a key exists. Returns None when disabled / there are no events / the call fails."""
        if not (self.enabled and self.trace_enabled):
            return None
        events = (session or {}).get("events") or []
        by_id = {e["id"]: e for e in events if isinstance(e, dict) and "id" in e}
        if not by_id:
            return None
        system = (
            "You build the `trace` for a CueBench coding-session page: a chronological timeline "
            "of the 4-8 most notable moments, so a manager or the operator can read it and see "
            "exactly what happened — where time went, what helped, what cost them. It is strictly "
            "FACTUAL.\n"
            "\n"
            "You are given a PRE-EXTRACTED, TIMESTAMPED list of events from one session (plus its "
            "duration and totals). Every event already happened and carries an integer `id`. Pick "
            "the most notable events and label them. You MUST reference each by its `id` and you "
            "must NOT invent anything beyond what the event states. Do NOT output a time — code "
            "stamps the authoritative time from the id.\n"
            "\n"
            "OUTPUT: a JSON array ONLY (start with [ , end with ] , no prose, no markdown fences). "
            'Each entry: {"id":<int from the list>,"sig":"good|warn|info",'
            '"label":"3-6 words naming the specific event","detail":"one sentence grounded in '
            'that event\'s facts"}.\n'
            "\n"
            "EVENT KINDS (what each id means):\n"
            "- open    = the operator's FIRST message. Notable always. good if it names a file/"
            "function/success-criteria; warn if it's vague enough the agent must guess.\n"
            "- prompt  = a later operator message (a steer or redirect). good if it cleanly "
            "redirects (especially right after a loop); info if routine.\n"
            "- context = distinct files read before the first edit (files, n_files, span). info; "
            "only worth an entry if the build was large or slow.\n"
            "- loop    = the agent re-ran the EXACT same call `count` times across `turns` on "
            "`target`. count>=3 with no operator redirect after it is a warn.\n"
            "- edit    = `file` was edited `edits` times. An edit to a file unrelated to the "
            "opening task is a warn (name the file); otherwise usually skip.\n"
            "- verify  = a test/typecheck/lint/build command ran (`cmd`). good or info.\n"
            "- model   = the model used. info (warn only if clearly mismatched to the work).\n"
            "- close   = how it ended: ended_on 'operator' = the operator acted last; 'agent' = "
            "it closed on agent output. verify_after_last_edit=false with edits_total>0 means the "
            "final change was never tested in-session — a warn.\n"
            "\n"
            "sig: 'good' = a smart operator move (tight opening, early redirect of a loop, a "
            "verify run that caught something). 'warn' = something that hurt the session (a loop "
            "with no redirect, a vague opening that forced guessing, an out-of-scope file touch, "
            "closing on agent output with an untested change). 'info' = a neutral fact (model, a "
            "notable context build).\n"
            "\n"
            "label — name the ACTUAL thing, never a category. Wrong: 'Loop detected' / Right: "
            "'5-turn loop on schema.sql'. Wrong: 'Good opening' / Right: 'Task scoped to one "
            "function'. Wrong: 'Context assembled' / Right: 'Read billing.py + 11 others'.\n"
            "detail — one sentence citing the event's real facts (the file, the count, the turn "
            "range, the operator's actual words, the command). Specificity test: if this sentence "
            "could describe a DIFFERENT session, make it more specific until it couldn't.\n"
            "\n"
            "CAMERA, NOT A JUDGE — never use scoring or competency words: no "
            "'delegation/description/discernment/diligence', no 'good/poor quality', no "
            "'strong/weak', no scores. Describe what happened; the scoring layer draws conclusions.\n"
            "\n"
            "COUNT: 4-8 entries, in any order (code sorts them by time). Fewer if fewer notable "
            "moments — an empty array [] is better than padding. Priority: the most impactful warn "
            "first, then the smartest good move, then 1-2 info anchors (model, notable context "
            "build), then any other genuinely notable good/warn.\n"
            "BANNED: restating the label in the detail; naming a competency axis; describing the "
            "scoring system instead of the session; generic praise with no cited fact."
        )
        user = json.dumps({k: session.get(k) for k in ("duration", "totals", "events")},
                          ensure_ascii=False)[:16000]
        raw = self._complete(system, user, max_tokens=900)
        data = _extract_json(raw) if raw else None
        if not isinstance(data, list) or not data:
            return None
        clean, seen = [], set()
        for it in data[:8]:
            if not isinstance(it, dict):
                continue
            _id = it.get("id")
            if _id not in by_id or _id in seen:         # must point at a REAL event, once each
                continue
            seen.add(_id)
            label = str(it.get("label", "")).strip()
            if not label:
                continue
            clean.append({
                "t": by_id[_id].get("t", "?"),          # TIME FROM CODE — never the model
                "sig": it.get("sig") if it.get("sig") in ("good", "warn", "info") else "info",
                "label": label[:48],
                "detail": str(it.get("detail", "")).strip()[:320],
            })
        clean.sort(key=lambda e: _t_to_secs(e["t"]))    # chronological
        return clean or None

    # NOTE: specificity() was REMOVED from BYOK — it now runs locally via the on-device encoder
    # (cuebench_classify.EmbeddingTypeClassifier.specificity). No OpenAI embeddings call.

    # --- the ONLY thing the model ever sees: derived numbers ------------------------
    @staticmethod
    def _metrics_block(metrics: dict, vectors: dict, quality: dict | None = None) -> str:
        """Serialize the numbers-only payload sent to the model. NO raw prompts/paths/cmds.
        The optional `quality` verdict (state/zone/label/definition — all derived numbers
        and fixed labels, no raw content) is included so generations stay coherent with the
        headline zone."""
        block = {"axis_scores_0_100": {a: vectors.get(a) for a in AXES}, "signals": metrics}
        if quality:
            block["quality"] = {k: quality.get(k) for k in
                                ("state", "zone", "label", "definition", "urgent")
                                if quality.get(k) is not None}
        return json.dumps(block, ensure_ascii=False, sort_keys=True)

    # --- the OPT-IN content the model sees ONLY in prompt-informed insights mode ----
    @staticmethod
    def _prompts_block(prompts: list[str] | None,
                       max_chars: int = 3500, max_prompts: int = 20) -> str:
        """Format the operator's prompts for prompt-informed insights (opt-in only). Sent ONLY
        to the user's own BYOK endpoint, NEVER stored in the payload. Capped (count + chars) so
        a long session can't balloon the request or its cost."""
        lines, budget = [], max_chars
        for i, p in enumerate((prompts or [])[:max_prompts], 1):
            p = (p or "").strip()
            if not p:
                continue
            chunk = f"[P{i}] {p[:600]}"
            lines.append(chunk[:budget])
            budget -= len(chunk)
            if budget <= 0:
                break
        return ("OPERATOR PROMPTS (the operator's own words this session — coach on how the "
                "work was scoped, sequenced, and phrased; paraphrase, don't quote at length):\n"
                + "\n".join(lines))
