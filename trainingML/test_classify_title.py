#!/usr/bin/env python3
"""
test_classify_title.py — the local title framework + task classification (no model, no network).

Covers the contract the title rebuild must keep:
  * <= 5 words, never the verbatim prompt;
  * canonical/extracted verb + the salient noun phrase;
  * command pastes and vague prompts fall back to the task label;
  * no usable prompt -> None (so the agent falls back to a date).

Run:  python3 -m pytest test_classify_title.py -q
"""
from __future__ import annotations
import cuebench_classify as c


def _title(prompt, task_type="feature"):
    return c.compose_title(task_type, c.TYPES[task_type]["label"], prompt)


def test_title_is_short_and_not_verbatim():
    p = "fix the off-by-one in pagination offset when page=0"
    t = _title(p, "bug_fix")
    assert t == "Fix pagination offset"
    assert len(t.split()) <= 5
    assert t.lower() != p.lower()                 # NOT the verbatim prompt


def test_extracts_verb_and_salient_phrase():
    assert _title("read PLAN.md and implement Milestone 0, then stop for review") == "Implement Milestone 0"
    assert _title("integrate Stripe webhook signature validation", "api_integration").startswith("Integrate Stripe")
    assert _title("reduce p95 latency on the /api/data endpoint", "performance") == "Reduce p95 latency"
    assert _title("ok so the ui of the cuebench menu bar app is pretty bad. lets make it nicer looking.",
                  "ui") == "Update cuebench menu bar"


def test_every_title_is_at_most_five_words():
    prompts = [
        "add a 300ms debounce to the search handler in search.js",
        "write unit tests for validateToken covering expired and malformed tokens",
        "investigate why the checkout endpoint intermittently returns 503",
        "extract the billing logic into a separate service layer",
        "hey claude im going to start making the backend for my ai fluency grading platform",
    ]
    for p in prompts:
        t = _title(p)
        assert t and len(t.split()) <= 5, (p, t)


def test_proper_nouns_preserved_generic_lowercased():
    # identifiers / proper nouns keep case; ordinary words are lowercased after the verb
    assert "Stripe" in _title("integrate Stripe webhook signature validation", "api_integration")
    t = _title("fix the off-by-one in pagination offset when page=0", "bug_fix")
    assert "pagination" in t and "Pagination" not in t


def test_command_paste_falls_back_to_label():
    assert _title("mkdir -p tests/golden\ncp ~/x.py .", "config") == "Configuration / setup"
    assert _title("git diff HEAD~3 -- src/", "code_review") == "Code review / audit"


def test_no_prompt_returns_none():
    assert _title("") is None
    assert _title("   ") is None
    assert c.compose_title("feature", "Feature build", None) is None


def test_ordinal_resolved_from_referenced_doc():
    # "implement Milestone 0" is opaque; with the referenced doc (e.g. recovered from the
    # transcript's read tool_results) the title names what M0 actually IS.
    doc = "## Milestones\n### M0 — Scaffolding\nTasks: init repo...\n### M3 — Git + Config collectors\nTasks:..."
    assert c.compose_title("feature", "Feature build",
                           "read PLAN.md and implement Milestone 0, then stop", doc_text=doc) == "Implement Scaffolding"
    assert c.compose_title("feature", "Feature build",
                           "now implement Milestone 3", doc_text=doc).startswith("Implement Git")
    # without the doc it falls back to the literal ordinal (no resolution invented)
    assert c.compose_title("feature", "Feature build",
                           "read PLAN.md and implement Milestone 0") == "Implement Milestone 0"
    # the plain instruction line in the doc must NOT be mistaken for the heading
    only_instruction = "say: read PLAN.md and implement Milestone 0, then stop for review"
    assert c.resolve_ordinal("implement Milestone 0", only_instruction) is None


def test_classify_emits_title_taskType_label():
    out = c.classify({"prompts": ["fix the flaky timeout in the auth test suite"], "n_edits": 1, "n_tools": 5})
    assert out["taskType"] == "test_fix"
    assert out["title"] and len(out["title"].split()) <= 5
    assert out["label"] == "Test fix"
