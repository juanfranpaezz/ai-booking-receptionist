"""Regression guard for the dry-run summary (review finding A1, 2026-09-26).

The dry run is a CI step. Its closing summary used to be a hard-coded string that
said the injection attempt "was blocked" even when the Layer-1 guard never fired,
so a broken guard left the step green and the console claiming a block that did
not happen. These tests pin the summary (and the exit code) to the guard's real
result, in both directions.
"""

from __future__ import annotations

import pytest

from booking_receptionist import __main__ as cli
from booking_receptionist import agent as agent_module

GUARD_EVENT_LINE = "[guard]  injection detected"
BLOCK_CLAIM = "injection attempt was blocked"


def test_dry_run_claims_block_when_guard_fires(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.run_dry_run()
    out = capsys.readouterr().out
    assert GUARD_EVENT_LINE in out
    assert BLOCK_CLAIM in out
    assert exit_code == 0


def test_dry_run_does_not_claim_block_when_guard_disabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Disable Layer 1 exactly where the agent reads it.
    monkeypatch.setattr(agent_module, "is_injection_attempt", lambda _message: False)
    exit_code = cli.run_dry_run()
    out = capsys.readouterr().out
    assert GUARD_EVENT_LINE not in out  # precondition: the guard really did not fire
    assert BLOCK_CLAIM not in out
    assert exit_code != 0
