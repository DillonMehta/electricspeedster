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

Content-aware titles are the ONE opt-in exception, exposed as Generator.content_title():
it is fed the session's FIRST PROMPT (not the metrics) and returns a short title that names
the task. The raw prompt is sent ONLY to the user's own BYOK endpoint (exactly like
specificity); only the short generated title leaves for the payload — the raw prompt itself
never does. Everything else (insights / trace / _metrics_block) stays numbers-only: do NOT
add raw prompts to the metrics dict here.

Config (env, per machine — never hardcode a key, never ship a default key)
--------------------------------------------------------------------------
  CUEBENCH_BYOK_KEY        the API key (absent -> generation OFF)
  CUEBENCH_BYOK_PROVIDER   "anthropic" | "openai"  (optional; else inferred from key/model)
  CUEBENCH_BYOK_MODEL      model id (optional; cheap per-provider default otherwise)
  CUEBENCH_TRACE           "1"/"true"/"on" to enable the optional trace (OFF by default)

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
                 specificity_enabled: bool | None = None):
        self.key = (key if key is not None else os.environ.get("CUEBENCH_BYOK_KEY")) or None
        env_model = os.environ.get("CUEBENCH_BYOK_MODEL")
        self.provider = (provider or os.environ.get("CUEBENCH_BYOK_PROVIDER")
                         or _infer_provider(self.key, env_model))
        self.model = model or env_model or _DEFAULT_MODEL.get(self.provider, "")
        self.trace_enabled = (_truthy(os.environ.get("CUEBENCH_TRACE"))
                              if trace_enabled is None else bool(trace_enabled))
        # specificity = a Description cross-check via OpenAI EMBEDDINGS. It is the ONE signal
        # that must see raw prompt TEXT, so it is OFF by default (opt-in) and OpenAI-only.
        self.specificity_enabled = (_truthy(os.environ.get("CUEBENCH_SPECIFICITY"))
                                    if specificity_enabled is None else bool(specificity_enabled))
        self.calls = 0          # count of BYOK API calls actually issued (dry-run proof)

    @property
    def enabled(self) -> bool:
        return bool(self.key)

    def _specificity_active(self) -> bool:
        return bool(self.enabled and self.specificity_enabled and self.provider == "openai")

    def status(self) -> str:
        if not self.enabled:
            return "off (no CUEBENCH_BYOK_KEY)"
        return (f"{self.provider}:{self.model} trace={'on' if self.trace_enabled else 'off'} "
                f"specificity={'on' if self._specificity_active() else 'off'}")

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

    # --- the three generations (numbers in, neutral text out) -----------------------
    def title(self, metrics: dict, vectors: dict, quality: dict | None = None) -> str | None:
        """A short (3-6 word) NEUTRAL session title describing behaviour, not subject."""
        system = _NEUTRALITY + (
            " Write a SHORT session title of 3 to 6 words that characterises HOW the session "
            "went (the working style and follow-through), based only on the metrics. No "
            "quotes, no trailing punctuation, no specifics. Output ONLY the title text."
        ) + _coherence_clause(quality)
        raw = self._complete(system, self._metrics_block(metrics, vectors, quality), max_tokens=40)
        if not raw:
            return None
        title = raw.strip().strip('"').splitlines()[0][:60].strip()
        return title or None

    def content_title(self, first_prompt: str | None,
                      max_words: int = 6, max_chars: int = 60) -> str | None:
        """A SHORT, CONTENT-AWARE title that NAMES THE TASK, generated from the session's
        FIRST PROMPT. This is the explicit opt-in the module docstring describes: the raw
        first prompt is sent ONLY to the user's own BYOK endpoint (like specificity), and only
        the short generated title is returned — the raw prompt itself never enters the payload.
        Returns None when generation is off / there is no prompt / the call fails, so the
        caller can fall back to a non-content label."""
        if not (self.enabled and first_prompt):
            return None
        system = (
            "You write an extremely short title for a coding session. Given the operator's "
            f"first prompt, reply with ONLY a title of at most {max_words} words and at most "
            f"{max_chars} characters. Name the task, not filler. No quotes, no trailing "
            "punctuation, no emoji."
        )
        raw = self._complete(system, first_prompt[:2000], max_tokens=24)
        if not raw:
            return None
        title = raw.strip().strip('"').strip("'").splitlines()[0].strip()
        if len(title) > max_chars:
            title = title[:max_chars].rsplit(" ", 1)[0].strip() or title[:max_chars].strip()
        return title.rstrip(" .") or None

    def insights(self, metrics: dict, vectors: dict, quality: dict | None = None) -> dict | None:
        """{'strengths':[{axis,point}...], 'improvements':[{axis,point}...]} or None.

        2-3 of each; every item tied to one of the four axes; content-neutral. The tone
        is kept coherent with the headline quality zone (see _coherence_clause): a
        'Critical'/'Needs attention' verdict must not read as mild encouragement.
        """
        system = _NEUTRALITY + (
            " Produce per-session coaching as STRICT JSON: "
            '{"strengths":[{"axis":"<one of delegation|description|discernment|diligence>",'
            '"point":"<one neutral sentence about driving behaviour>"}],'
            '"improvements":[{"axis":"...","point":"..."}]}. '
            "Give 2-3 strengths and 2-3 improvements. Tie each item to the axis its score/"
            "signals most support. Describe behaviour, never subject matter. Output ONLY JSON."
        ) + _coherence_clause(quality)
        raw = self._complete(system, self._metrics_block(metrics, vectors, quality), max_tokens=700)
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
                        clean.append({"axis": axis, "point": point[:240]})
            out[bucket] = clean[:3]
        if not out.get("strengths") and not out.get("improvements"):
            return None
        return out

    def trace(self, metrics: dict, vectors: dict, quality: dict | None = None) -> list | None:
        """Optional neutral action-narrative. Only generated when trace is enabled AND a key
        exists (the caller already gates on trace_enabled; this re-checks defensively)."""
        if not (self.enabled and self.trace_enabled):
            return None
        system = _NEUTRALITY + (
            " Produce a concise timeline of the session's WORKING BEHAVIOUR as STRICT JSON: "
            'an array of 4-8 objects {"t":"MM:SS","sig":"good|warn|info","label":"<=6 words",'
            '"detail":"one neutral sentence about behaviour"}. Base it only on the metrics '
            "(verification, commits, loops, reverts, churn, scores). Never name files, code, "
            "or commands. Output ONLY the JSON array."
        ) + _coherence_clause(quality)
        raw = self._complete(system, self._metrics_block(metrics, vectors, quality), max_tokens=900)
        data = _extract_json(raw) if raw else None
        if not isinstance(data, list) or not data:
            return None
        clean = []
        for it in data[:8]:
            if not isinstance(it, dict):
                continue
            clean.append({
                "t": str(it.get("t", "00:00"))[:8],
                "sig": it.get("sig") if it.get("sig") in ("good", "warn", "info") else "info",
                "label": str(it.get("label", ""))[:40],
                "detail": str(it.get("detail", ""))[:240],
            })
        return clean or None

    def specificity(self, prompts: list[str]) -> int | None:
        """0-100 prompt-specificity cross-check (Description axis), via OpenAI EMBEDDINGS.

        This is the ONE signal that must see raw prompt TEXT — it measures how specific the
        prompts are, which cannot be derived from counts. The text is sent ONLY to the user's
        own BYOK embeddings endpoint; only the resulting 0-100 NUMBER ever enters the payload.
        OFF unless explicitly enabled (opt-in), and OpenAI-only — Anthropic has no embeddings
        API. Returns None when disabled / not OpenAI / no usable prompts / call fails."""
        if not self._specificity_active():
            return None
        self.calls += 1
        try:
            import openai
            import cuebench_signals as sig
            val = sig.specificity(prompts, openai.OpenAI(api_key=self.key))
            return None if val is None else int(round(val))
        except Exception as e:
            print(f"  [gen] specificity skipped ({e!r}); degrading gracefully", file=sys.stderr)
            return None

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
