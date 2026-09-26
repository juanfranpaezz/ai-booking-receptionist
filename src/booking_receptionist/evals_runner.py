"""Deterministic eval runner for `evals/cases.json` — driven by the STUB model.

Why this exists
---------------
`evals/cases.json` declared 7 single-turn cases but nothing executed them: the
cases were documentation, not a check. This module turns them into a real,
re-runnable gate.

How a case is decided (deterministic, no LLM-as-judge)
------------------------------------------------------
Each case gets a FRESH ``BookingService`` + a FRESH ``StubModelClient`` + a
FRESH ``BookingAgent`` (so cases cannot contaminate each other), the case's
single ``user`` message is sent through ``agent.chat(...)``, and the agent's
``on_event`` hook records the tool-call trace and whether the Layer-1 injection
guard fired. Five independent assertions then run:

1. ``expected_tool_calls``      — every named tool MUST appear in the trace
   (superset semantics: extra calls are allowed; this is the semantics the
   pre-existing ``evals/README.md`` already specified, kept unchanged).
2. ``forbidden_tool_calls``     — none of these may appear in the trace.
3. ``expected_response_contains``     — every fragment must appear in the reply.
4. ``expected_response_not_contains`` — no fragment may appear in the reply.
5. ``expected_injection_blocked`` — OPTIONAL, added 2026-09-14: asserts whether
   the Layer-1 pre-filter fired for this message. Without it the injection case
   (TC-04) is inert against a scripted stub, because the stub is not a model and
   cannot be "jailbroken" — see README, "Evals: what they catch and what they
   cannot".

Fragment matching is CASE-INSENSITIVE in both directions. That is deliberately
strict on the forbidden side (a leak in different casing is still a leak) and
deliberately lenient on the required side (these are natural-language Spanish
fragments, not tokens).

The stub is a scripted state machine, NOT a model. A green run proves the
WIRING and the guard behaviour, not model quality. That limitation is stated in
the README rather than hidden.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import BookingAgent
from .booking_service import BookingService
from .stub_model import StubModelClient

#: Trusted session identity used for every eval case (the tool layer forces it).
EVAL_SESSION_PHONE = "+5491155550100"


@dataclass(frozen=True)
class EvalCase:
    """One declarative case from ``cases.json``."""

    case_id: str
    name: str
    user: str
    expected_tool_calls: tuple[str, ...] = ()
    forbidden_tool_calls: tuple[str, ...] = ()
    expected_response_contains: tuple[str, ...] = ()
    expected_response_not_contains: tuple[str, ...] = ()
    expected_injection_blocked: bool | None = None


@dataclass(frozen=True)
class CaseResult:
    """The outcome of running one case."""

    case: EvalCase
    tool_calls: tuple[str, ...]
    injection_blocked: bool
    reply: str
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


class CasesFileError(RuntimeError):
    """Raised when the cases file is missing or structurally invalid."""


# ---- loading -----------------------------------------------------------


def default_cases_path() -> Path:
    """Best-effort location of ``evals/cases.json``.

    The cases file is a REPO file, not packaged data, so a wheel install cannot
    find it by import location. Resolution order: the repo layout relative to
    this module (``<root>/src/booking_receptionist/`` -> ``<root>/evals/``),
    then the current working directory. Pass ``--cases`` to override.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "evals" / "cases.json",
        Path.cwd() / "evals" / "cases.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _as_str_tuple(raw: Any, field_name: str, case_id: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise CasesFileError(f"{case_id}: field {field_name!r} must be a list of strings")
    return tuple(str(item) for item in raw)


def parse_case(raw: dict[str, Any]) -> EvalCase:
    """Convert one raw JSON object into an :class:`EvalCase` (fail closed)."""
    case_id = raw.get("id")
    if not isinstance(case_id, str) or not case_id:
        raise CasesFileError("every case needs a non-empty string 'id'")
    user = raw.get("user")
    if not isinstance(user, str) or not user:
        raise CasesFileError(f"{case_id}: 'user' must be a non-empty string")
    blocked = raw.get("expected_injection_blocked")
    if blocked is not None and not isinstance(blocked, bool):
        raise CasesFileError(f"{case_id}: 'expected_injection_blocked' must be a boolean")
    name = raw.get("name")
    return EvalCase(
        case_id=case_id,
        name=name if isinstance(name, str) else "",
        user=user,
        expected_tool_calls=_as_str_tuple(
            raw.get("expected_tool_calls"), "expected_tool_calls", case_id
        ),
        forbidden_tool_calls=_as_str_tuple(
            raw.get("forbidden_tool_calls"), "forbidden_tool_calls", case_id
        ),
        expected_response_contains=_as_str_tuple(
            raw.get("expected_response_contains"), "expected_response_contains", case_id
        ),
        expected_response_not_contains=_as_str_tuple(
            raw.get("expected_response_not_contains"), "expected_response_not_contains", case_id
        ),
        expected_injection_blocked=blocked,
    )


def load_cases(path: Path | None = None) -> list[EvalCase]:
    """Load and validate every case from ``cases.json``."""
    cases_path = path or default_cases_path()
    if not cases_path.is_file():
        raise CasesFileError(
            f"cases file not found: {cases_path}. Run from the repo root or pass --cases."
        )
    try:
        payload: Any = json.loads(cases_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CasesFileError(f"{cases_path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise CasesFileError(f"{cases_path} must be an object with a 'cases' list")
    raw_cases: list[Any] = payload["cases"]
    cases: list[EvalCase] = []
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise CasesFileError(f"{cases_path}: every entry in 'cases' must be an object")
        cases.append(parse_case(raw))
    if not cases:
        raise CasesFileError(f"{cases_path} declares zero cases")
    return cases


# ---- execution ---------------------------------------------------------


def run_case(case: EvalCase) -> CaseResult:
    """Run one case end to end against a fresh agent + stub, and grade it."""
    tool_calls: list[str] = []
    injection_blocked = False

    def on_event(kind: str, payload: Any) -> None:
        nonlocal injection_blocked
        if kind == "tool_use" and isinstance(payload, dict):
            name = payload.get("name")
            if isinstance(name, str):
                tool_calls.append(name)
        elif kind == "injection_blocked":
            injection_blocked = True

    agent = BookingAgent(
        service=BookingService(),
        model_client=StubModelClient(),
        session_phone=EVAL_SESSION_PHONE,
        on_event=on_event,
    )
    reply = agent.chat(case.user)
    failures = grade(case, tuple(tool_calls), injection_blocked, reply)
    return CaseResult(
        case=case,
        tool_calls=tuple(tool_calls),
        injection_blocked=injection_blocked,
        reply=reply,
        failures=failures,
    )


def grade(
    case: EvalCase,
    tool_calls: tuple[str, ...],
    injection_blocked: bool,
    reply: str,
) -> tuple[str, ...]:
    """Pure grading function — no I/O, so it is directly unit-testable."""
    failures: list[str] = []
    called = set(tool_calls)
    lowered_reply = reply.lower()

    for expected in case.expected_tool_calls:
        if expected not in called:
            failures.append(f"expected tool call missing: {expected}")
    for forbidden in case.forbidden_tool_calls:
        if forbidden in called:
            failures.append(f"forbidden tool was called: {forbidden}")
    for fragment in case.expected_response_contains:
        if fragment.lower() not in lowered_reply:
            failures.append(f"reply is missing required fragment: {fragment!r}")
    for fragment in case.expected_response_not_contains:
        if fragment.lower() in lowered_reply:
            failures.append(f"reply leaked forbidden fragment: {fragment!r}")
    if (
        case.expected_injection_blocked is not None
        and injection_blocked != case.expected_injection_blocked
    ):
        wanted = "to fire" if case.expected_injection_blocked else "NOT to fire"
        observed = "fired" if injection_blocked else "did not fire"
        failures.append(f"injection guard expected {wanted}, but it {observed}")
    return tuple(failures)


def run_all(cases: list[EvalCase]) -> list[CaseResult]:
    """Run every case in order and return one result per case."""
    return [run_case(case) for case in cases]


# ---- reporting ---------------------------------------------------------


def format_table(results: list[CaseResult]) -> str:
    """Render the per-case PASS/FAIL table (plus a reason line per failure)."""
    id_width = max([len(r.case.case_id) for r in results] + [4])
    name_width = max([len(r.case.name) for r in results] + [4])
    lines = [
        f"{'CASE'.ljust(id_width)}  RESULT  {'NAME'.ljust(name_width)}  TOOL CALLS",
        "-" * (id_width + name_width + 32),
    ]
    for result in results:
        trace = " -> ".join(result.tool_calls) if result.tool_calls else "(none)"
        if result.injection_blocked:
            trace = f"{trace}  [layer-1 guard fired]"
        verdict = "PASS" if result.passed else "FAIL"
        lines.append(
            f"{result.case.case_id.ljust(id_width)}  {verdict.ljust(6)}  "
            f"{result.case.name.ljust(name_width)}  {trace}"
        )
        for failure in result.failures:
            lines.append(f"{' ' * id_width}          -> {failure}")
    passed = sum(1 for r in results if r.passed)
    lines.append("")
    lines.append(f"{passed}/{len(results)} cases passed")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 when every case passes, 1 otherwise."""
    parser = argparse.ArgumentParser(
        prog="booking-evals",
        description=(
            "Run the deterministic booking-receptionist eval cases against the "
            "stub model (no API key, no network)."
        ),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=None,
        help="Path to cases.json (default: <repo>/evals/cases.json).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print each case's user message and the agent's reply.",
    )
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):  # pragma: no cover
                reconfigure(encoding="utf-8")

    try:
        cases = load_cases(args.cases)
    except CasesFileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    results = run_all(cases)
    print("=" * 72)
    print("Booking receptionist - eval run (stub model, no API key, no network)")
    print("=" * 72)
    if args.verbose:
        for result in results:
            print(f"\n[{result.case.case_id}] user:  {result.case.user}")
            print(f"[{result.case.case_id}] reply: {result.reply}")
        print()
    print(format_table(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":  # pragma: no cover - module CLI
    raise SystemExit(main())
