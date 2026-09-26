# Evals

A small, deterministic eval set for the booking agent, mirroring the production
design philosophy: **assert on the tool-call trace and on string fragments of
the reply — no LLM-as-judge.** An LLM grading an LLM is easy to fool into
self-congratulation; concrete "did it call `create_appointment` or not" /
"did the reply contain `París`" checks are honest and cheap.

## What's here

- `cases.json` — 7 single-turn cases (`TC-01`…`TC-07`): happy-path availability
  lookup, service inquiry, temporal ambiguity, prompt-injection resilience at the
  *model* level, out-of-scope redirect, cancellation-needs-lookup, and Argentine
  slang comprehension. Each case declares `expected_tool_calls`,
  `forbidden_tool_calls`, `expected_response_contains`, and
  `expected_response_not_contains`. TC-04 also carries the optional boolean
  `expected_injection_blocked` (added 2026-09-14) — see "Why TC-04 needs an
  extra assertion" below.
- `run.py` — a zero-install entry point (`python evals/run.py`). The
  implementation lives in the package at
  `src/booking_receptionist/evals_runner.py`, so it is type-checked under
  `mypy --strict` and installed as the `booking-evals` console script.

## How to run it

```bash
python evals/run.py            # PASS/FAIL table; exit 0 all pass, 1 any fail, 2 bad cases file
python evals/run.py --verbose  # also prints each case's user message and reply
python -m pytest               # the same 7 cases as parametrized tests, plus the grader's own tests
```

What the runner does per case: build a **fresh** `BookingService` +
`StubModelClient` + `BookingAgent`, send `user` through `agent.chat(...)`,
capture the `tool_use` events and the `injection_blocked` event via the
`on_event` hook, then assert that the captured tool names ⊇
`expected_tool_calls`, are disjoint from `forbidden_tool_calls`, that the reply
satisfies the contains / not-contains checks (case-insensitively), and that the
Layer-1 guard fired iff `expected_injection_blocked` says so.

To evaluate against the **real** model instead, swap `StubModelClient()` for
`AnthropicModelClient()` with `ANTHROPIC_API_KEY` set. That path is deliberately
never run here or in CI: it costs money, it is non-deterministic, and it would
put a secret in CI.

## Why TC-04 needs an extra assertion

Against a scripted stub, TC-04 would pass **even with the Layer-1 pre-filter
deleted** — the stub is not a model, so it cannot be jailbroken; it would simply
return its generic greeting, which leaks nothing the case forbids. The case
would have looked green while testing nothing. `expected_injection_blocked: true`
is what makes it bite, and the planted-defect check in the top-level README is
what proves it bites.

## Two stub branches were added for these cases

`stub_model.py` had no scripted response for a cancellation request (TC-06) or
for a slang availability question (TC-07), so those two cases could not run at
all. Two branches were added on 2026-09-14, each tagged `EVAL EXTENSION` in the
source with the case it serves. This is worth saying out loud: **part of what
these two cases assert is the stub's own keyword routing, not the model's
comprehension.**

## Honest limitations (deliberately not hidden)

- **Single-turn only** — no multi-turn dialogue state is exercised here.
- **Injection coverage is illustrative, and stub-bounded** — `TC-04` tests *one*
  vector, and against the stub it proves only that the Layer-1 pre-filter fires
  and the model is never called. It says nothing about model-level resilience to
  a novel jailbreak.
- **`TC-07` proves routing, not comprehension** — it shows a slang phrasing
  reaches `get_available_slots` in the stub's keyword table.
- **`TC-02` and `TC-03` effectively assert only their forbidden sets** — the stub
  always follows `get_services` with `get_available_slots`, and the tool-name
  rule is a superset check, so their positive expectations pass trivially.
- **No latency or cost measurement.** Regression + CI gating now exist
  (`.github/workflows/ci.yml`), but only over the stub path.
- **Spanish (Argentine) only.**

This is a v0.1 baseline, sized to prove the pattern — not a production eval suite.
