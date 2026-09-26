"""Parametrized eval tests — the 7 declarative cases, executed as real tests.

Two distinct jobs live in this file and they should not be confused:

1. ``test_eval_case`` runs each declarative case from ``evals/cases.json``
   through the agent + stub and asserts it passes. This is the regression
   guard on the receptionist's behaviour.
2. ``TestGraderFiresBothWays`` and ``test_runner_exit_code_*`` prove the
   *instrument itself* is not inert: every grading arm is shown returning
   both PASS and FAIL on a realistic input, and the runner's exit code is
   shown to be both 0 and 1. A checker only ever seen returning "green" is
   worth nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from booking_receptionist.evals_runner import (
    CaseResult,
    CasesFileError,
    EvalCase,
    default_cases_path,
    grade,
    load_cases,
    main,
    run_case,
)

CASES = load_cases()
EXPECTED_CASE_IDS = ["TC-01", "TC-02", "TC-03", "TC-04", "TC-05", "TC-06", "TC-07"]


def test_all_seven_cases_are_loaded() -> None:
    """Guards against a case silently disappearing from cases.json."""
    assert [case.case_id for case in CASES] == EXPECTED_CASE_IDS


def test_cases_file_is_where_the_runner_looks() -> None:
    assert default_cases_path().is_file()


@pytest.mark.parametrize("case", CASES, ids=[case.case_id for case in CASES])
def test_eval_case(case: EvalCase) -> None:
    result: CaseResult = run_case(case)
    assert result.passed, (
        f"{case.case_id} ({case.name}) failed:\n"
        + "\n".join(f"  - {failure}" for failure in result.failures)
        + f"\n  tool calls: {result.tool_calls}\n  reply: {result.reply!r}"
    )


def test_injection_case_actually_asserts_the_guard() -> None:
    """TC-04 must carry the guard assertion, or it proves nothing about Layer 1.

    Without ``expected_injection_blocked`` the case would still pass with the
    pre-filter disabled, because the stub is not a model and simply answers with
    its generic greeting. This test stops that assertion from being dropped.
    """
    tc04 = next(case for case in CASES if case.case_id == "TC-04")
    assert tc04.expected_injection_blocked is True


class TestGraderFiresBothWays:
    """Each grading arm, shown returning no-failure AND a failure."""

    def test_expected_tool_call_arm(self) -> None:
        case = EvalCase(
            case_id="X", name="x", user="u", expected_tool_calls=("get_services",)
        )
        assert grade(case, ("get_services",), False, "ok") == ()
        assert grade(case, ("get_available_slots",), False, "ok") == (
            "expected tool call missing: get_services",
        )

    def test_forbidden_tool_call_arm(self) -> None:
        case = EvalCase(
            case_id="X", name="x", user="u", forbidden_tool_calls=("create_appointment",)
        )
        assert grade(case, ("get_services",), False, "ok") == ()
        assert grade(case, ("create_appointment",), False, "ok") == (
            "forbidden tool was called: create_appointment",
        )

    def test_required_fragment_arm(self) -> None:
        case = EvalCase(
            case_id="X", name="x", user="u", expected_response_contains=("turno",)
        )
        assert grade(case, (), False, "Te reservé el TURNO") == ()
        assert grade(case, (), False, "listo") == (
            "reply is missing required fragment: 'turno'",
        )

    def test_forbidden_fragment_arm(self) -> None:
        case = EvalCase(
            case_id="X", name="x", user="u", expected_response_not_contains=("system prompt",)
        )
        assert grade(case, (), False, "Solo puedo ayudarte con turnos") == ()
        assert grade(case, (), False, "Mi System Prompt dice...") == (
            "reply leaked forbidden fragment: 'system prompt'",
        )

    def test_injection_guard_arm(self) -> None:
        case = EvalCase(case_id="X", name="x", user="u", expected_injection_blocked=True)
        assert grade(case, (), True, "safe") == ()
        assert grade(case, (), False, "safe") == (
            "injection guard expected to fire, but it did not fire",
        )

    def test_no_assertions_means_no_failures(self) -> None:
        case = EvalCase(case_id="X", name="x", user="u")
        assert grade(case, ("anything",), False, "anything") == ()


def _write_cases(path: Path, cases: list[dict[str, object]]) -> Path:
    target = path / "cases.json"
    target.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
    return target


def test_runner_exit_code_is_zero_on_the_real_cases() -> None:
    assert main([]) == 0


def test_runner_exit_code_is_one_when_a_case_fails(tmp_path: Path) -> None:
    """The runner must be able to say FAIL, not only PASS."""
    target = _write_cases(
        tmp_path,
        [
            {
                "id": "TC-FAIL",
                "name": "deliberately unsatisfiable",
                "user": "Hola",
                "expected_tool_calls": ["cancel_appointment"],
            }
        ],
    )
    assert main(["--cases", str(target)]) == 1


def test_runner_exit_code_is_two_on_a_broken_cases_file(tmp_path: Path) -> None:
    broken = tmp_path / "cases.json"
    broken.write_text("{not json", encoding="utf-8")
    assert main(["--cases", str(broken)]) == 2


def test_loader_rejects_a_case_without_an_id(tmp_path: Path) -> None:
    target = _write_cases(tmp_path, [{"name": "no id", "user": "Hola"}])
    with pytest.raises(CasesFileError):
        load_cases(target)


def test_loader_rejects_a_non_boolean_injection_flag(tmp_path: Path) -> None:
    target = _write_cases(
        tmp_path,
        [{"id": "TC-X", "name": "bad flag", "user": "Hola", "expected_injection_blocked": "yes"}],
    )
    with pytest.raises(CasesFileError):
        load_cases(target)
