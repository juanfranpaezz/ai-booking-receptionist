"""Regression guard for the dry-run summary (review finding A1, 2026-09-26).

The dry run is a CI step. Its closing summary used to be a hard-coded string that
said the injection attempt "was blocked" even when the Layer-1 guard never fired,
so a broken guard left the step green and the console claiming a block that did
not happen. These tests pin the summary (and the exit code) to the guard's real
result, in both directions.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from booking_receptionist import __main__ as cli
from booking_receptionist import agent as agent_module
from booking_receptionist.injection_guard import is_injection_attempt
from booking_receptionist.stub_model import StubModelClient

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


def test_dry_run_does_not_claim_block_when_agent_does_not_stop(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The guard fires and reports the event, but the agent ignores its own guard and
    # still sends the message to the model (the `return SAFE_REPLY` is missing).
    def chat_ignoring_guard(self: agent_module.BookingAgent, user_message: str) -> str:
        if is_injection_attempt(user_message):
            self._on_event("injection_blocked", user_message)
        self.messages.append({"role": "user", "content": user_message})
        return self._run_loop()

    monkeypatch.setattr(agent_module.BookingAgent, "chat", chat_ignoring_guard)
    exit_code = cli.run_dry_run()
    out = capsys.readouterr().out
    assert GUARD_EVENT_LINE in out  # precondition: the guard really did fire
    assert BLOCK_CLAIM not in out
    assert "passed to the model" in out
    assert exit_code != 0


# Review finding F1 (2026-09-27): the summary used to decide "reached the model" by
# looking for the exact injection string among the user texts the model was sent, so
# an agent that forwarded the injection in any other form still got "blocked before
# the model was ever called" and exit 0. The claim is now checked literally (zero
# model calls during the injection turn). Each test below plants one such regression
# in the agent, first proves with a spy that the model really received the injection
# text, then requires the summary to refuse the block claim and exit non-zero.


def _spy_on_model_payloads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every system + messages payload the stub model is really called with."""
    payloads: list[str] = []
    original = StubModelClient.create_message

    def spy(self: StubModelClient, **kwargs: Any) -> dict[str, Any]:
        payloads.append(
            json.dumps(
                {"system": kwargs["system"], "messages": kwargs["messages"]},
                ensure_ascii=False,
            )
        )
        return original(self, **kwargs)

    monkeypatch.setattr(StubModelClient, "create_message", spy)
    return payloads


def _assert_no_block_claim(out: str, exit_code: int, payloads: list[str]) -> None:
    assert GUARD_EVENT_LINE in out  # precondition: the guard really did fire
    # precondition: the model really received the injection text, in some form
    assert any(cli.DRY_RUN_INJECTION in payload for payload in payloads)
    assert BLOCK_CLAIM not in out
    assert "Dry run FAILED" in out
    assert exit_code != 0


def test_dry_run_does_not_claim_block_when_injection_is_forwarded_annotated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # "Flag it and let the model handle it": the guard fires, the agent tags the
    # message and still sends it on.
    def chat_annotating_injection(self: agent_module.BookingAgent, user_message: str) -> str:
        if is_injection_attempt(user_message):
            self._on_event("injection_blocked", user_message)
            user_message = "[posible inyeccion] " + user_message
        self.messages.append({"role": "user", "content": user_message})
        return self._run_loop()

    monkeypatch.setattr(agent_module.BookingAgent, "chat", chat_annotating_injection)
    payloads = _spy_on_model_payloads(monkeypatch)
    exit_code = cli.run_dry_run()
    _assert_no_block_claim(capsys.readouterr().out, exit_code, payloads)


def test_dry_run_does_not_claim_block_when_every_user_message_is_prefixed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Metadata framing on every turn: no user text the model receives is ever
    # byte-equal to what the customer typed.
    def chat_prefixing_every_turn(self: agent_module.BookingAgent, user_message: str) -> str:
        if is_injection_attempt(user_message):
            self._on_event("injection_blocked", user_message)
        self.messages.append({"role": "user", "content": "[cliente] " + user_message})
        return self._run_loop()

    monkeypatch.setattr(agent_module.BookingAgent, "chat", chat_prefixing_every_turn)
    payloads = _spy_on_model_payloads(monkeypatch)
    exit_code = cli.run_dry_run()
    _assert_no_block_claim(capsys.readouterr().out, exit_code, payloads)


def test_dry_run_does_not_claim_block_when_injection_is_moved_into_system_prompt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The guard fires, the agent moves the flagged text into the system prompt "as
    # context" and sends a neutral placeholder as the user turn.
    def chat_moving_injection_to_system(self: agent_module.BookingAgent, user_message: str) -> str:
        if is_injection_attempt(user_message):
            self._on_event("injection_blocked", user_message)
            self._system.append({"type": "text", "text": "Mensaje filtrado: " + user_message})
            user_message = "[mensaje retenido por el filtro]"
        self.messages.append({"role": "user", "content": user_message})
        return self._run_loop()

    monkeypatch.setattr(agent_module.BookingAgent, "chat", chat_moving_injection_to_system)
    payloads = _spy_on_model_payloads(monkeypatch)
    exit_code = cli.run_dry_run()
    _assert_no_block_claim(capsys.readouterr().out, exit_code, payloads)
