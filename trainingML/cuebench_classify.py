#!/usr/bin/env python3
"""
cuebench_classify.py — deterministic task-type + title for a CueBench session
=============================================================================
ADDITIVE. This never touches the scorer or the frozen digest. It reads the same
`inputs` dict that cuebench_signals.build_inputs() produces and emits:

    {"taskType": <key>, "label": <human label>, "title": <str>,
     "confidence": float, "method": "rule" | "embedding"}

Nothing here is AI-generated at runtime:
  * title   = a short (<=5 word) task title via LOCAL keyphrase extraction — a
              canonical/extracted verb + the single most salient noun phrase. NOT the
              verbatim prompt (see the TITLE FRAMEWORK section). Deterministic, no model.
  * taskType= a priority-ordered keyword rule layer (transparent, auditable)
              with a LOCAL-EMBEDDING fallback for the ambiguous remainder.

The embedding fallback reuses the encoder ModelScorer already loads (roberta-base)
and classifies by nearest TYPE CENTROID, where the centroids are built from the
example phrases in the published taxonomy (TYPES below). No labeled data, no
second model download, no network call. Deterministic given fixed weights.

    from cuebench_classify import classify, EmbeddingTypeClassifier
    res = classify(inputs)                          # rules only (no torch needed)
    emb = EmbeddingTypeClassifier.from_scorer(my_scorer)   # reuse loaded encoder
    res = classify(inputs, embedder=emb)            # rules + embedding fallback
"""
from __future__ import annotations
import re

# ============================================================================
# 1. THE 18 TYPES — key, label, title verb-pattern, and example anchors.
#    `anchors` are lifted from the published taxonomy doc; they double as the
#    centroid training phrases for the embedding fallback (no separate dataset).
# ============================================================================
TYPES = {
 "bug_fix":        {"label": "Bug fix",
    "anchors": ["fix null return in getUserById", "repair broken pagination on /orders",
                "fix off-by-one in date range filter"]},
 "debug":          {"label": "Debug / investigate",
    "anchors": ["investigate intermittent 503 on checkout", "trace why cron job silently exits",
                "find source of memory leak in worker"]},
 "feature":        {"label": "Feature build",
    "anchors": ["add rate limiting to /api/session", "build CSV export for the reports tab",
                "implement webhook retry logic"]},
 "refactor":       {"label": "Refactor",
    "anchors": ["extract billing logic into service layer", "simplify auth middleware conditionals",
                "rename user to operator across codebase"]},
 "test_writing":   {"label": "Test writing",
    "anchors": ["write unit tests for validateToken", "add integration tests for /api/employees",
                "cover edge cases in date parsing module"]},
 "test_fix":       {"label": "Test fix",
    "anchors": ["fix flaky timeout in auth test suite", "update snapshot tests after UI refactor",
                "repair broken seed data in e2e setup"]},
 "api_integration":{"label": "API integration",
    "anchors": ["integrate Stripe webhook signature validation", "add Slack notification on deploy failure",
                "implement GitHub OAuth callback"]},
 "database":       {"label": "Database / migration",
    "anchors": ["add index on sessions.employee_id", "write migration to backfill org slugs",
                "optimize N+1 query in employee list"]},
 "performance":    {"label": "Performance optimization",
    "anchors": ["reduce p95 latency on /api/data", "cut bundle size for mobile dashboard",
                "lower token cost in scoring prompt"]},
 "security":       {"label": "Security fix",
    "anchors": ["add SQL injection protection to search", "rotate and re-encrypt stored API keys",
                "fix missing auth check on admin endpoint"]},
 "config":         {"label": "Configuration / setup",
    "anchors": ["set up GitHub Actions deploy to Azure", "configure ESLint rules for the project",
                "add environment variables for staging"]},
 "documentation":  {"label": "Documentation",
    "anchors": ["write API reference for /api/session", "update README with new env vars",
                "add JSDoc to auth middleware"]},
 "code_review":    {"label": "Code review / audit",
    "anchors": ["review PR add payment retry logic", "audit new auth middleware for security",
                "check migration script for data loss risk"]},
 "explanation":    {"label": "Explanation / learning",
    "anchors": ["explain how the resolveOrg middleware works", "walk through the session scoring pipeline",
                "understand the Socket.io event flow"]},
 "search":         {"label": "Search / exploration",
    "anchors": ["find all callers of getDb", "locate where API keys are validated",
                "list all endpoints that require auth"]},
 "ui":             {"label": "UI / frontend",
    "anchors": ["add copy button to API key field", "fix mobile layout on employee detail",
                "build collapsible import panel for admin"]},
 "script":         {"label": "Script / automation",
    "anchors": ["script to backfill employee initials", "write CSV import parser for bulk upload",
                "build log analysis script for error patterns"]},
 "architecture":   {"label": "Architecture / design",
    "anchors": ["design multi-tenant data isolation approach", "plan migration from REST to Socket.io",
                "evaluate caching strategy for /api/data"]},
}

# ============================================================================
# 2. RULE LAYER — priority-ordered. MORE SPECIFIC patterns first, because the
#    cheap keywords (fix / add) overlap many types. Each entry:
#       (type_key, compiled_pattern, confidence)
#    First match on the FIRST prompt wins. Confidence < FALLBACK_THRESHOLD hands
#    off to the embedding layer (or to the signal-based default if none).
# ============================================================================
FALLBACK_THRESHOLD = 0.7
# Minimum top1-top2 cosine margin for the embedding fallback to be trusted over a sensible
# default. v1 ESTIMATE — TUNE against the labeled corpus (judgments.json). Below it, the
# embedding result is treated as too ambiguous and `_signal_default` is used instead.
EMBED_MARGIN_MIN = 0.03

# Prompt-specificity (Description cross-check), computed LOCALLY via the encoder (NO external
# embeddings API). For each prompt: nearest-SPECIFIC-pole cosine minus nearest-VAGUE-pole
# cosine; averaged, then mapped to 0-100 over [SPEC_MARGIN_LO, SPEC_MARGIN_HI]. The band was
# calibrated to roberta-base: novel specific prompts land ~+0.23, neutral ~+0.05, vague ~-0.22,
# so a symmetric ±0.25 band puts specific~95 / neutral~59 / vague~7. Retune if the encoder changes.
SPEC_MARGIN_LO = -0.25
SPEC_MARGIN_HI = 0.25
SPECIFIC_POLES = ["add a 300ms debounce to the search handler in search.js",
    "fix the off-by-one in pagination offset when page=0",
    "return 404 instead of 500 when the record id is missing",
    "add a unit test asserting the parser rejects inputs over 1MB"]
VAGUE_POLES = ["make it nice", "fix it", "make it better", "clean it up", "improve this",
    "do the thing", "make it work", "handle it", "just make it good"]

_RULES = [
 # --- test fix BEFORE test writing BEFORE bug fix (all share "fix"/"test") ---
 ("test_fix",        r"\b(flaky|broken|failing|update snapshot)\b.*\b(tests?|specs?|suite|snapshots?)\b|"
                     r"\b(fix|repair|unbreak)\b.*\b(tests?|specs?|suite|snapshots?|fixture|seed data)\b", 0.9),
 ("test_writing",    r"\b(write|add|cover|create)\b.*\b(unit test|integration test|e2e test|tests?|specs?|coverage|edge cases?)\b|"
                     r"\btest coverage\b", 0.9),
 # --- security before bug fix (a security "fix" is its own type) ---
 ("security",        r"\b(security|vulnerab\w*|sql injection|injection|xss|csrf|sanitiz\w*|auth(oriz\w*)? check|"
                     r"missing auth|secrets?|encrypt\w*|re-encrypt|harden\w*|exploit|permission check)\b", 0.9),
 # --- database / migration ---
 ("database",        r"\b(migration|migrate|backfill|schema|add (an? )?index|n\+1|slow quer\w*|optimi\w* .*quer\w*|"
                     r"sql quer\w*|orm|seed the (db|database))\b", 0.85),
 # --- api integration ---
 ("api_integration",r"\b(integrat\w*|webhook|oauth|sdk|stripe|slack|twilio|sendgrid|third[- ]party|"
                     r"signature validation|callback url|api client)\b", 0.85),
 # --- performance ---
 ("performance",     r"\b(optimi\w*|reduce|cut|lower|speed up|faster|throughput|p95|p99|latency|"
                     r"bundle size|memory (use|footprint)|token cost)\b", 0.8),
 # --- refactor ---
 ("refactor",        r"\b(refactor\w*|extract|simplif\w*|rename|clean ?up|de-?duplicat\w*|restructur\w*|"
                     r"tidy|reorganiz\w*|consolidat\w*)\b", 0.85),
 # --- documentation ---
 ("documentation",   r"\b(document\w*|docs?|readme|changelog|jsdoc|docstring|api reference|"
                     r"add comments?|inline comment)\b", 0.85),
 # --- config / setup ---
 ("config",          r"\b(set ?up|configure|config\w*|ci/?cd|github actions|ci pipeline|build pipeline|"
                     r"deploy(ment)? pipeline|eslint|prettier|env(ironment)? var\w*|dockerfile|"
                     r"docker-compose|deploy\w*|build system|tooling)\b", 0.8),
 # --- ui / frontend ---
 ("ui",              r"\b(ui|frontend|component|layout|css|styl\w*|button|modal|dropdown|panel|sidebar|"
                     r"navbar|tooltip|responsive|mobile (layout|view)|animation|theme|tailwind)\b", 0.8),
 # --- code review / audit (review/audit must be the ACTION, not an incidental word like
 #     "stop for review"; require it leading or with a review object) ---
 ("code_review",     r"^\s*(review|audit)\b|\b(review (this|the|my|our|these)\s+\w*\s?(pr|diff|change|"
                     r"migration|code|implementation)|audit .*\b(security|code|auth|migration)\b|"
                     r"look over (this|the|my)|sanity[- ]check)\b", 0.8),
 # --- explanation / learning ---
 ("explanation",     r"\b(explain|understand|how does|how do|walk (me )?through|what does|"
                     r"clarify how|help me understand)\b", 0.8),
 # --- search / exploration ---
 ("search",          r"\b(find all|locate|where (is|are|does)|which file|list all|search for|"
                     r"who (calls|uses)|grep for)\b", 0.8),
 # --- architecture / design (anchor design/plan/evaluate as the ACTION; bare "plan" must
 #     not match a filename like PLAN.md, nor "approach"/"strategy" as incidental nouns) ---
 ("architecture",    r"^\s*(design|plan|evaluate|architect)\b|"
                     r"\b(design (a|an|the)\s|plan (a|an|the|for|to|out|the migration)|"
                     r"evaluate (a|an|the|whether|options|the)|architecture|caching strategy|"
                     r"trade-?offs?|decide between|propose (an?|the) (design|approach|architecture))\b", 0.75),
 # --- script / automation ---
 ("script",          r"\b(script to|one-?off|automat\w*|parser for|batch (job|process)|"
                     r"data pipeline|log analysis)\b", 0.75),
 # --- debug / investigate (symptom-first; before generic bug fix) ---
 ("debug",           r"\b(investigat\w*|debug|trace|diagnos\w*|root cause|why (is|does|do|are)|"
                     r"find (the )?(source|cause)|intermittent|reproduce|silently (exits?|fails?))\b", 0.8),
 # --- bug fix (known defect, generic 'fix') ---
 ("bug_fix",         r"\b(fix|repair|broken|off-by-one|returns? null|incorrect|wrong|crash\w*|"
                     r"error in|bug in|not working)\b", 0.6),
 # --- feature build (new capability; lowest-specificity verbs, last) ---
 ("feature",         r"\b(add|build|implement|create|introduce|support for|new (feature|endpoint|page))\b", 0.55),
]
_RULES = [(k, re.compile(p, re.I), c) for (k, p, c) in _RULES]


# ============================================================================
# 3. TITLE FRAMEWORK — a short (<=5 word) task title via LOCAL keyphrase extraction.
#    NOT the verbatim prompt: a canonical/extracted VERB + the single most salient
#    noun phrase. Research-grounded (RAKE-style stopword-delimited candidates, made
#    noun-phrase-ish + identifier-aware per PatternRank/KeyBERT; YAKE-style position
#    weighting), but fully deterministic and dependency-free — no model, no network.
#    Pipeline: clean -> drop shell/command pastes -> strip conversational preamble ->
#    pick the action VERB (the prompt's own, else a per-type canonical) -> score
#    clause-local candidate phrases (identifier > content >> filler; content-only
#    multiword bonus; earlier favored) -> compose "<Verb> <phrase>". Falls back to the
#    task label for command pastes / vague prompts, None when there is no prompt.
# ============================================================================
TITLE_MAX_WORDS = 5

_T_STOP = set("""a an the this that these those to of in on for and or but with without from into onto at
by as is are was were be been being it its do does did so then than just only also very really too my your
our their his her i you we they me us them he she not no yes ok okay here there now about when while where
which who what why how all any some more most up over out against between have has had will would should
could can may might must get got even still back want need make made before after fully doing following
around each both off down per via run show give tell let go come take put keep set start""".split())
_T_FILLER = set("""code codebase file files thing things stuff app apps application project repo repository
implementation bit part way ways new current existing whole entire good nice cool better best simple example
lot bunch thanks pretty bad sucks looking anything everything something real actually basically please pls
senior dev guy kinda sorta maybe separate reusable proper covering including regarding""".split())
_T_PREAMBLE = re.compile(
    r"^(?:ok(?:ay)?|so|well|hey|hi|hello|claude|alright|right|now|next|actually|basically|cool|thanks|"
    r"please|pls|lets?|let'?s|lets?\s+go|go\s+back\s+to|back\s+to|first|then|um+|uh+|yo|dude|man|"
    r"i\s*(?:'?m)?\s*(?:want(?:ing)?|need(?:ing)?|would\s+like|wanna|gonna|gotta|going\s+to|guess|"
    r"think|just)?\s*(?:to|you\s+to)?|we\s+(?:need|want|should|gotta)\s+to)\b[\s,:.\-]*", re.I)
# action verb -> canonical display form (extracted from the prompt when present)
_T_VERB_FORMS = {
    "fix":"Fix","repair":"Fix","unbreak":"Fix","debug":"Debug","investigate":"Investigate","trace":"Trace",
    "diagnose":"Diagnose","add":"Add","build":"Build","implement":"Implement","create":"Create","make":"Build",
    "write":"Write","refactor":"Refactor","simplify":"Simplify","extract":"Extract","rename":"Rename",
    "clean":"Clean up","optimize":"Optimize","optimise":"Optimize","reduce":"Reduce","speed":"Speed up",
    "improve":"Improve","update":"Update","remove":"Remove","delete":"Remove","drop":"Remove",
    "integrate":"Integrate","connect":"Connect","migrate":"Migrate","review":"Review","audit":"Audit",
    "explain":"Explain","find":"Find","locate":"Find","document":"Document","configure":"Configure",
    "setup":"Set up","test":"Test","cover":"Test","harden":"Harden","secure":"Secure","validate":"Validate",
    "design":"Design","plan":"Plan","evaluate":"Evaluate","polish":"Polish","move":"Move","copy":"Copy"}
# pure action verbs (never a noun) -> excluded from candidate phrases so the object stays clean
_T_PURE_VERBS = {"implement","integrate","investigate","refactor","optimize","optimise","debug","diagnose",
    "configure","harden","simplify","migrate","repair","unbreak","rename","validate","add","fix","build",
    "create","remove","delete","improve","reduce","make","write","review","audit","explain","find",
    "document","design","evaluate","polish","move","copy","connect"}
_T_NAV = {"read","open","look","go","check","see","view","use","start"}
# per-taskType canonical verb (fallback when the prompt has no clear leading action verb)
TASK_VERB = {"bug_fix":"Fix","debug":"Investigate","feature":"Add","refactor":"Refactor","test_writing":"Test",
    "test_fix":"Fix","api_integration":"Integrate","database":"Update","performance":"Optimize",
    "security":"Harden","config":"Configure","documentation":"Document","code_review":"Review",
    "explanation":"Explain","search":"Find","ui":"Update","script":"Script","architecture":"Design"}
_T_IDENT = re.compile(r"[a-z][A-Z]|[_/]|\.[A-Za-z]{1,4}\b|\d|[A-Z]{2,}")   # camel/snake/path/.ext/digit/ACRONYM
_T_FNAME = re.compile(r"[./_]")
_T_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/=\-]*")
_T_CMD = re.compile(r"^\s*(?:sudo\s+)?(?:mkdir|cd|ls|cp|mv|rm|touch|chmod|export|git|npm|pip|python3?|"
                    r"node|yarn|pnpm|brew|curl|wget|docker|kubectl|\./)\b")

def _t_clean(text: str) -> str:
    t = text or ""
    t = re.sub(r"<[^>]+>", " ", t)                       # xml-ish tags (ide_selection, leftover reminders)
    t = re.sub(r"```.*?```", " ", t, flags=re.DOTALL)    # fenced code
    t = re.sub(r"#region.*", " ", t, flags=re.DOTALL)    # pasted code regions
    t = re.sub(r"`[^`]*`", " ", t)                       # inline code
    t = re.sub(r"https?://\S+", " ", t)                  # urls
    return re.sub(r"\s+", " ", t).strip()

def _t_is_ident(w: str) -> bool:
    return bool(_T_IDENT.search(w))

def _t_is_low(w: str) -> bool:
    """Low-value word: generic filler or an -ly adverb (intermittently, really, ...)."""
    lw = w.lower()
    return lw in _T_FILLER or (lw.endswith("ly") and len(lw) > 5)

def _t_strip_preamble(t: str) -> str:
    for _ in range(5):                                   # peel stacked preambles ("ok so lets")
        n = _T_PREAMBLE.sub("", t)
        if n == t:
            break
        t = n
    return t

def _t_find_verb(text: str) -> str | None:
    """First ACTION verb near the front (skipping filenames + navigation verbs like 'read'),
    so 'read PLAN.md then implement X' -> Implement. None if none found in the first ~12 tokens."""
    n = 0
    for m in _T_TOKEN.finditer(text):
        tok = m.group(0); n += 1
        if n > 12:
            break
        if _T_FNAME.search(tok):                         # filename/identifier, never a verb
            continue
        w = re.sub(r"[^A-Za-z']", "", tok).lower()
        if w in _T_VERB_FORMS:
            return _T_VERB_FORMS[w]
    return None

def _t_best_phrase(body: str) -> list:
    """The single highest-salience candidate phrase across the body's clauses."""
    best, best_s = [], 0.0
    for ci, clause in enumerate(re.split(r"[.!?;,\n]|\s+then\s+|\s+-\s+", body)):
        if not clause.strip():
            continue
        cur, pi = [], 0
        def flush(cur, pi):
            nonlocal best, best_s
            ph = cur[:4]
            while ph and _t_is_low(ph[0]) and not _t_is_ident(ph[0]): ph = ph[1:]
            while ph and _t_is_low(ph[-1]) and not _t_is_ident(ph[-1]): ph = ph[:-1]
            if not ph:
                return
            s, content = 0.0, 0
            for w in ph:
                if _t_is_ident(w): s += 2.0; content += 1
                elif _t_is_low(w): s += 0.15
                else: s += 1.0; content += 1
            if content == 0:
                return
            s *= (1.0 + 0.25 * (content - 1))            # multiword bonus on CONTENT words only
            s *= (1.0 - 0.08 * ci) * (1.0 - 0.05 * pi)   # earlier clause & earlier phrase favored
            if s > best_s:
                best_s, best = s, ph
        for tok in _T_TOKEN.findall(clause):
            w = tok.lower().strip(".")
            if w in _T_STOP or w in _T_PURE_VERBS or w in _T_NAV:
                if cur: flush(cur, pi); pi += 1; cur = []
                continue
            cur.append(tok.strip(".,"))
            if len(cur) >= 4: flush(cur, pi); pi += 1; cur = []
        if cur: flush(cur, pi)
    return best

def _t_case(w: str) -> str:
    if _t_is_ident(w) or any(c.isupper() for c in w[1:]) or w[:1].isupper():
        return w                                         # preserve identifiers / acronyms / proper nouns
    return w.lower()

# Ordinal references ("Milestone 0", "Phase 3", "Step 2", "M0") name a section defined in a
# plan/spec doc — opaque on their own. resolve_ordinal looks the section up in the doc the
# session referenced (recovered from the transcript's read tool_results, or from disk) and
# returns its real descriptor, so "implement Milestone 0" -> "Scaffolding".
_ORDINAL = re.compile(r"\b(milestone|phase|step|task|stage|part|chapter|section)\s*#?\s*(\d+)\b", re.I)

def has_ordinal(text: str | None) -> bool:
    return bool(text and _ORDINAL.search(text))

def resolve_ordinal(prompt: str, doc_text: str, max_words: int = 3) -> list | None:
    """If `prompt` names an ordinal section, find that HEADING in `doc_text` and return its
    descriptor words. Requires a heading-style separator (— : -) after the ordinal so the
    plain instruction line ('implement Milestone 0, then…') is NOT mistaken for the heading."""
    if not doc_text:
        return None
    m = _ORDINAL.search(prompt)
    if not m:
        return None
    word, num = m.group(1).lower(), m.group(2)
    abbr = word[0]                                       # milestone->m, phase->p, step->s …
    pat = re.compile(rf"#{{0,6}}\s*(?:{word}\s*{num}|{abbr}{num})\b\s*[:：—–\-]\s*([A-Za-z][^\n\r\t]{{1,55}})", re.I)
    hit = pat.search(doc_text)
    if not hit:
        return None
    desc = re.split(r"\bTasks\b|\bAcceptance\b|\bGoal\b", hit.group(1))[0]
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z+/]*", desc) if w.lower() not in _T_STOP][:max(1, max_words)]
    return words or None

def compose_title(task_type: str, task_label: str, prompt: str | None,
                  max_words: int = TITLE_MAX_WORDS, doc_text: str | None = None) -> str | None:
    """Short task title (<= max_words). Returns the task label for command pastes / vague
    prompts, and None when there is no usable prompt (so the caller can fall back).

    When the prompt names an ordinal section ('Milestone 0') AND `doc_text` (the referenced
    plan/spec, recovered from the transcript or disk) is provided, the opaque ordinal is
    replaced by the section's real descriptor: 'implement Milestone 0' -> 'Implement Scaffolding'."""
    raw = _t_clean(prompt or "")
    if not raw:
        return None
    if _T_CMD.match(raw):                                # pasted shell -> the label is the honest title
        return task_label
    body = _t_strip_preamble(raw) or raw
    verb = _t_find_verb(body) or TASK_VERB.get(task_type, "Update")
    budget = max_words - len(verb.split())
    if doc_text:                                         # resolve 'Milestone 0' -> its real descriptor
        desc = resolve_ordinal(raw, doc_text, max_words=budget)
        if desc:
            return (f"{verb} " + " ".join(_t_case(w) for w in desc)).strip(" .,:-")
    phrase = _t_best_phrase(body)
    obj = [w for w in phrase if not _t_is_low(w)][:budget] or phrase[:budget]
    real = [w for w in obj if _t_is_ident(w) or len(w) > 2]
    if not real:                                         # nothing meaningful extracted -> label
        return task_label
    title = f"{verb} " + " ".join(_t_case(w) for w in obj)
    title = re.sub(r"^(\w+)\s+\1\b", r"\1", title, flags=re.I)   # dedup verb echoed in object
    return title.strip(" .,:-")


# ============================================================================
# 4. SIGNAL-AWARE DEFAULT — when rules are weak and no embedder is available,
#    use the deterministic signals to pick a sane default instead of guessing.
# ============================================================================
def _signal_default(inputs: dict) -> tuple[str, float]:
    edits = int(inputs.get("n_edits", 0) or 0)
    commits = int(inputs.get("n_commits", 0) or 0)
    tools = int(inputs.get("n_tools", 0) or 0)
    # No code written and nothing committed -> a read/understand session.
    if edits == 0 and commits == 0:
        # lots of tool calls but no edits => exploring the codebase
        return ("explanation" if tools >= 8 else "search", 0.4)
    return ("feature", 0.35)   # code was written, type unclear -> assume new work


# ============================================================================
# 5. EMBEDDING FALLBACK — nearest type centroid, using the LOCAL encoder.
#    Reuse the encoder ModelScorer already loaded (from_scorer) so there is no
#    second model in memory (matters on the 17GB box). torch imported lazily.
# ============================================================================
class EmbeddingTypeClassifier:
    def __init__(self, tok, enc, device: str = "cpu", maxlen: int = 64):
        import torch  # lazy
        self._torch = torch
        self.tok, self.enc, self.device, self.maxlen = tok, enc, device, maxlen
        self.enc.eval()
        self.keys = list(TYPES.keys())
        # one centroid per type = mean of its (normalized) anchor embeddings
        cents = []
        for k in self.keys:
            v = self._embed(TYPES[k]["anchors"]).mean(0)
            cents.append(v / (v.norm() + 1e-9))
        self.centroids = torch.stack(cents)            # [18, hidden]
        # specificity poles (embedded once) — for the local Description cross-check
        self._spec_poles = self._embed(SPECIFIC_POLES)   # [ns, hidden] (normalized)
        self._vague_poles = self._embed(VAGUE_POLES)     # [nv, hidden] (normalized)

    @classmethod
    def from_scorer(cls, scorer, maxlen: int = 64):
        """Reuse a loaded ModelScorer's tokenizer + encoder. Zero extra weights."""
        return cls(scorer.tok, scorer.model.enc, getattr(scorer, "device", "cpu"), maxlen)

    @classmethod
    def from_base(cls, base: str = "roberta-base", device: str = "cpu", maxlen: int = 64):
        from transformers import AutoTokenizer, AutoModel
        return cls(AutoTokenizer.from_pretrained(base),
                   AutoModel.from_pretrained(base), device, maxlen)

    def _embed(self, texts):
        torch = self._torch
        enc = self.tok(list(texts), truncation=True, max_length=self.maxlen,
                       padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.enc(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        h, m = out.last_hidden_state, enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)        # mean-pool, mask-aware
        return pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-9)

    def classify(self, text: str) -> tuple[str, float, float]:
        """Return (type_key, top_cosine, margin). `margin` = top1 - top2 cosine: how
        DECISIVELY this prompt resembles one type over the next. With these encoders the
        absolute cosines bunch together, so the margin — not the raw cosine — is the real
        signal of confidence. The caller gates on it (see EMBED_MARGIN_MIN)."""
        v = self._embed([text])[0]
        sims = (self.centroids @ v)                            # cosine (both normalized)
        order = sims.argsort(descending=True)
        i = int(order[0])
        top = float(sims[i])
        second = float(sims[int(order[1])]) if len(order) > 1 else 0.0
        return self.keys[i], top, top - second

    def specificity(self, prompts) -> int | None:
        """0-100 prompt-specificity (Description cross-check), computed LOCALLY via this
        encoder — NO external embeddings API, no raw text off-device. For each usable prompt:
        nearest-SPECIFIC-pole cosine minus nearest-VAGUE-pole cosine; averaged, then mapped to
        0-100 over [SPEC_MARGIN_LO, SPEC_MARGIN_HI]. Returns None when there is no usable prompt."""
        texts = [p for p in (prompts or []) if p and len(p.split()) >= 2]
        if not texts:
            return None
        P = self._embed(texts)
        s = (P @ self._spec_poles.T).max(dim=1).values
        v = (P @ self._vague_poles.T).max(dim=1).values
        margin = float((s - v).mean())
        val = (margin - SPEC_MARGIN_LO) / (SPEC_MARGIN_HI - SPEC_MARGIN_LO)
        return int(round(max(0.0, min(1.0, val)) * 100))


# ============================================================================
# 6. PUBLIC ENTRY — rules first, embedding fallback, signal default last.
# ============================================================================
def rule_classify(prompts) -> tuple[str | None, float, str | None]:
    """First matching rule on the first non-empty prompt. (key, conf, matched)."""
    first = next((p for p in (prompts or []) if p and p.strip()), "")
    for key, pat, conf in _RULES:
        m = pat.search(first)
        if m:
            return key, conf, m.group(0)
    return None, 0.0, None


def classify(inputs: dict, embedder: "EmbeddingTypeClassifier | None" = None,
             doc_text: str | None = None) -> dict:
    prompts = inputs.get("prompts", [])
    key, conf, matched = rule_classify(prompts)
    method = "rule"

    if key is None or conf < FALLBACK_THRESHOLD:
        used_embedding = False
        if embedder is not None:
            first = next((p for p in prompts if p and p.strip()), "")
            ekey, esim, emargin = embedder.classify(first)
            # Trust the embedding only when it DISCRIMINATES (clear top-1 vs top-2 margin).
            # A bunched, low-margin result is noise — don't let it override a sensible default.
            if emargin >= EMBED_MARGIN_MIN and (key is None or esim >= conf):
                key, conf, method = ekey, round(esim, 3), "embedding"
                used_embedding = True
        if key is None and not used_embedding:
            key, conf, method = (*_signal_default(inputs),)[:2] + ("signal_default",)

    # Title uses the FINAL taskType (its canonical verb is the fallback when the prompt has no
    # clear action verb) — short, keyphrase-extracted, never the verbatim prompt.
    first = next((p for p in prompts if p and p.strip()), "")
    title = compose_title(key, TYPES[key]["label"], first, doc_text=doc_text)
    return {"taskType": key, "label": TYPES[key]["label"], "title": title,
            "confidence": round(float(conf), 3), "method": method,
            "matched": matched}


# ============================================================================
# 7. Smoke test
# ============================================================================
if __name__ == "__main__":
    import json
    samples = [
        ["fix the off-by-one in pagination offset when page=0"],
        ["investigate why the checkout endpoint intermittently returns 503"],
        ["add a 300ms debounce to the search handler in search.js"],
        ["write unit tests for validateToken covering expired and malformed tokens"],
        ["fix the flaky timeout in the auth test suite"],
        ["explain how the resolveOrg middleware decides the tenant"],
        ["set up GitHub Actions to deploy to Azure on merge to main"],
        ["make it nicer"],   # vague -> falls through to signal/embedding
    ]
    for s in samples:
        print(f"{s[0][:55]:<57} -> ", json.dumps(classify({"prompts": s, "n_edits": 1, "n_tools": 5})))
