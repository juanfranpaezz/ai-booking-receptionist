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
  `expected_response_not_contains`.

## How you'd run it

A runner is intentionally left thin. To evaluate against the real model:

1. For each case, build a fresh `BookingAgent` (real `AnthropicModelClient`,
   `ANTHROPIC_API_KEY` set).
2. Send `user` through `agent.chat(...)`, capturing the `tool_use` events via the
   `on_event` hook.
3. Assert the captured tool names ⊇ `expected_tool_calls`, are disjoint from
   `forbidden_tool_calls`, and the reply text satisfies the contains/not-contains
   checks.

## Honest limitations (deliberately not hidden)

- **Single-turn only** — no multi-turn dialogue state is exercised here.
- **Injection coverage is illustrative** — `TC-04` tests *one* vector against the
  model's own resilience. The Layer-1 pre-filter (which blocks before the model)
  is covered by `tests/test_smoke.py`, not here.
- **No latency / cost / regression / CI gating.**
- **Spanish (Argentine) only.**

This is a v0.1 baseline, sized to prove the pattern — not a production eval suite.
