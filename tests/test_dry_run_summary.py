"""Regression guard for the dry-run summary (review finding A1, 2026-09-26).

The dry run is a CI step. Its closing summary used to be a hard-coded string that
said the injection attempt "was blocked" even when the Layer-1 guard never fired,
so a broken guard left the step green and the console claiming a block that did
not happen. These tests pin the summary (and the exit code) to the guard's real
result, in both directions.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from booking_receptionist import __main__ as cli
from booking_receptionist import agent as agent_module
from booking_receptionist.booking_service import BookingService
from booking_receptionist.injection_guard import SAFE_REPLY, is_injection_attempt
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


def test_dry_run_does_not_claim_block_when_injection_is_forwarded_reworded(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The guard fires and the agent forwards the message base64-encoded, so no text
    # match can see it: only the call made during the injection turn gives it away.
    encoded = base64.b64encode(cli.DRY_RUN_INJECTION.encode()).decode()

    def chat_forwarding_encoded(self: agent_module.BookingAgent, user_message: str) -> str:
        if is_injection_attempt(user_message):
            self._on_event("injection_blocked", user_message)
            user_message = encoded
        self.messages.append({"role": "user", "content": user_message})
        return self._run_loop()

    monkeypatch.setattr(agent_module.BookingAgent, "chat", chat_forwarding_encoded)
    payloads = _spy_on_model_payloads(monkeypatch)
    exit_code = cli.run_dry_run()
    out = capsys.readouterr().out
    assert GUARD_EVENT_LINE in out  # precondition: the guard really did fire
    assert any(encoded in payload for payload in payloads)  # precondition: it was sent
    assert not any(cli.DRY_RUN_INJECTION in payload for payload in payloads)
    assert BLOCK_CLAIM not in out
    assert "outside a later scripted turn" in out
    assert exit_code != 0


# Review finding (round 2, 2026-09-27): the claim above was only checked while the
# injection turn's chat() ran, so a model call carrying the injection LATER in the
# run (a later turn, a background thread, printing the reply, an exit hook) still
# got "blocked before the model was ever called" and exit 0. The claim now covers the
# whole run. Each plant below first proves the model really receives the injection
# (or, for calls after the run, that the call was really attempted).

FOLLOW_UP_TURN = "Gracias, eso es todo"


def _add_follow_up_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """A customer turn after the injection: the path a history leak needs."""
    monkeypatch.setattr(cli, "DRY_RUN_SCRIPT", [*cli.DRY_RUN_SCRIPT, FOLLOW_UP_TURN])


def _injection_call(agent: agent_module.BookingAgent, text: str) -> None:
    """A side call to the model (moderation, logging...) that carries `text`."""
    agent.client.create_message(
        model=agent.model,
        max_tokens=64,
        system=agent._system,
        tools=[],
        messages=[{"role": "user", "content": text}],
    )


def test_dry_run_does_not_claim_block_when_history_leaks_into_a_later_turn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The guard fires and the turn returns the safe reply with no model call, but the
    # message stays in the history that the next turn sends to the model.
    def chat_keeping_injection_in_history(
        self: agent_module.BookingAgent, user_message: str
    ) -> str:
        self.messages.append({"role": "user", "content": user_message})
        if is_injection_attempt(user_message):
            self._on_event("injection_blocked", user_message)
            return SAFE_REPLY
        return self._run_loop()

    monkeypatch.setattr(agent_module.BookingAgent, "chat", chat_keeping_injection_in_history)
    _add_follow_up_turn(monkeypatch)
    payloads = _spy_on_model_payloads(monkeypatch)
    exit_code = cli.run_dry_run()
    _assert_no_block_claim(capsys.readouterr().out, exit_code, payloads)


def test_dry_run_does_not_claim_block_when_reworded_history_leaks_into_a_later_turn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Same leak, but stored base64-encoded: no text match can see it. The injection
    # turn changed the history, and that alone voids the claim.
    encoded = base64.b64encode(cli.DRY_RUN_INJECTION.encode()).decode()

    def chat_keeping_encoded_injection(self: agent_module.BookingAgent, user_message: str) -> str:
        if is_injection_attempt(user_message):
            self._on_event("injection_blocked", user_message)
            self.messages.append({"role": "user", "content": "[retenido] " + encoded})
            return SAFE_REPLY
        self.messages.append({"role": "user", "content": user_message})
        return self._run_loop()

    monkeypatch.setattr(agent_module.BookingAgent, "chat", chat_keeping_encoded_injection)
    _add_follow_up_turn(monkeypatch)
    payloads = _spy_on_model_payloads(monkeypatch)
    exit_code = cli.run_dry_run()
    out = capsys.readouterr().out
    assert GUARD_EVENT_LINE in out  # precondition: the guard really did fire
    assert any(encoded in payload for payload in payloads)  # precondition: it leaked
    assert BLOCK_CLAIM not in out
    assert "changed the conversation history" in out
    assert exit_code != 0


def test_dry_run_does_not_claim_block_when_injection_is_stashed_for_a_later_turn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The injection turn leaves history and system prompt untouched, but the agent
    # keeps the message aside and sends it along with the next customer turn.
    stash: list[str] = []

    def chat_stashing_injection(self: agent_module.BookingAgent, user_message: str) -> str:
        if is_injection_attempt(user_message):
            self._on_event("injection_blocked", user_message)
            stash.append(user_message)
            return SAFE_REPLY
        content = "\n".join([*stash, user_message])
        stash.clear()
        self.messages.append({"role": "user", "content": content})
        return self._run_loop()

    monkeypatch.setattr(agent_module.BookingAgent, "chat", chat_stashing_injection)
    _add_follow_up_turn(monkeypatch)
    payloads = _spy_on_model_payloads(monkeypatch)
    exit_code = cli.run_dry_run()
    _assert_no_block_claim(capsys.readouterr().out, exit_code, payloads)


def test_dry_run_does_not_claim_block_when_a_background_thread_calls_the_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The guard fires and the turn returns at once, but a fire-and-forget worker
    # sends the message to the model a moment later, after the turn has ended.
    workers: list[threading.Thread] = []

    def chat_moderating_in_background(self: agent_module.BookingAgent, user_message: str) -> str:
        if is_injection_attempt(user_message):
            self._on_event("injection_blocked", user_message)

            def moderate() -> None:
                time.sleep(0.05)
                _injection_call(self, user_message)

            workers.append(threading.Thread(target=moderate, daemon=True))
            workers[-1].start()
            return SAFE_REPLY
        self.messages.append({"role": "user", "content": user_message})
        return self._run_loop()

    monkeypatch.setattr(agent_module.BookingAgent, "chat", chat_moderating_in_background)
    payloads = _spy_on_model_payloads(monkeypatch)
    exit_code = cli.run_dry_run()
    for worker in workers:
        worker.join(5)
    _assert_no_block_claim(capsys.readouterr().out, exit_code, payloads)


def test_dry_run_does_not_claim_block_when_a_later_turn_hands_it_to_another_thread(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The injection turn leaves history untouched; the next customer turn hands a
    # base64 copy to a worker thread that calls the model while that turn runs. Only
    # the scripted turn's own thread may call the model, so the worker gives it away.
    encoded = base64.b64encode(cli.DRY_RUN_INJECTION.encode()).decode()
    stash: list[str] = []

    def chat_with_worker_on_next_turn(self: agent_module.BookingAgent, user_message: str) -> str:
        if is_injection_attempt(user_message):
            self._on_event("injection_blocked", user_message)
            stash.append(encoded)
            return SAFE_REPLY
        if stash:
            worker = threading.Thread(target=_injection_call, args=(self, stash.pop()))
            worker.start()
            worker.join(5)
        self.messages.append({"role": "user", "content": user_message})
        return self._run_loop()

    monkeypatch.setattr(agent_module.BookingAgent, "chat", chat_with_worker_on_next_turn)
    _add_follow_up_turn(monkeypatch)
    payloads = _spy_on_model_payloads(monkeypatch)
    exit_code = cli.run_dry_run()
    out = capsys.readouterr().out
    assert GUARD_EVENT_LINE in out  # precondition: the guard really did fire
    assert any(encoded in payload for payload in payloads)  # precondition: it was sent
    assert BLOCK_CLAIM not in out
    assert "outside a later scripted turn" in out
    assert exit_code != 0


def test_dry_run_does_not_claim_block_when_printing_the_reply_calls_the_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # chat() returns with no model call, but the reply it returns calls the model
    # with the message when it is formatted for printing.
    def chat_returning_deferred_reply(self: agent_module.BookingAgent, user_message: str) -> str:
        if is_injection_attempt(user_message):
            self._on_event("injection_blocked", user_message)
            agent = self

            class DeferredReply(str):
                def __format__(self, format_spec: str) -> str:
                    _injection_call(agent, user_message)
                    return str.__format__(self, format_spec)

            return DeferredReply(SAFE_REPLY)
        self.messages.append({"role": "user", "content": user_message})
        return self._run_loop()

    monkeypatch.setattr(agent_module.BookingAgent, "chat", chat_returning_deferred_reply)
    payloads = _spy_on_model_payloads(monkeypatch)
    exit_code = cli.run_dry_run()
    _assert_no_block_claim(capsys.readouterr().out, exit_code, payloads)


def test_dry_run_never_sends_a_model_call_made_after_the_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A worker slower than the dry run's wait: its call comes after the summary was
    # decided. It must be refused, never sent, so the block claim stays true.
    # raising=False: on a version without the wait, the test must fail on behavior.
    monkeypatch.setattr(cli, "DRY_RUN_THREAD_WAIT_S", 0.05, raising=False)
    workers: list[threading.Thread] = []
    errors: list[BaseException] = []

    def chat_with_slow_worker(self: agent_module.BookingAgent, user_message: str) -> str:
        if is_injection_attempt(user_message):
            self._on_event("injection_blocked", user_message)

            def moderate() -> None:
                time.sleep(0.5)
                try:
                    _injection_call(self, user_message)
                except RuntimeError as exc:
                    errors.append(exc)

            workers.append(threading.Thread(target=moderate, daemon=True))
            workers[-1].start()
            return SAFE_REPLY
        self.messages.append({"role": "user", "content": user_message})
        return self._run_loop()

    monkeypatch.setattr(agent_module.BookingAgent, "chat", chat_with_slow_worker)
    payloads = _spy_on_model_payloads(monkeypatch)
    exit_code = cli.run_dry_run()
    for worker in workers:
        worker.join(5)
    out = capsys.readouterr().out
    assert workers  # precondition: the late worker really ran
    assert not any(cli.DRY_RUN_INJECTION in payload for payload in payloads)
    assert len(errors) == 1  # the late call was refused, not silently dropped
    assert BLOCK_CLAIM in out
    assert exit_code == 0


def test_dry_run_block_claim_holds_when_an_exit_hook_calls_the_model(
    tmp_path: Path,
) -> None:
    # An exit hook (atexit) calls the model with the message after the summary is
    # printed. Real interpreter exit is needed, so this runs in a child process.
    child = tmp_path / "exit_hook_plant.py"
    child.write_text(
        """
import atexit
import json

from booking_receptionist import __main__ as cli
from booking_receptionist import agent as agent_module
from booking_receptionist.injection_guard import SAFE_REPLY, is_injection_attempt
from booking_receptionist.stub_model import StubModelClient

received = []
original = StubModelClient.create_message


def spy(self, **kwargs):
    payload = json.dumps([kwargs["system"], kwargs["messages"]], ensure_ascii=False)
    received.append(cli.DRY_RUN_INJECTION in payload)
    return original(self, **kwargs)


StubModelClient.create_message = spy
# Registered first, so it runs last (exit hooks run in reverse order).
atexit.register(lambda: print(f"MODEL_CALLS_WITH_INJECTION={sum(received)}", flush=True))


def chat(self, user_message):
    if is_injection_attempt(user_message):
        self._on_event("injection_blocked", user_message)
        atexit.register(
            lambda: self.client.create_message(
                model=self.model,
                max_tokens=64,
                system=self._system,
                tools=[],
                messages=[{"role": "user", "content": user_message}],
            )
        )
        return SAFE_REPLY
    self.messages.append({"role": "user", "content": user_message})
    return self._run_loop()


agent_module.BookingAgent.chat = chat
raise SystemExit(cli.main(["--dry-run"]))
""",
        encoding="utf-8",
    )
    src_dir = str(Path(cli.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": src_dir, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, str(child)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=60,
        check=False,
    )
    match = re.search(r"MODEL_CALLS_WITH_INJECTION=(\d+)", result.stdout)
    assert match is not None, result.stderr
    received = int(match.group(1))
    assert GUARD_EVENT_LINE in result.stdout  # precondition: the guard really did fire
    # precondition: the exit hook really called the model client (sent or refused)
    assert received > 0 or "model call refused" in result.stderr
    # The block claim may only be printed if the model never got the message.
    assert BLOCK_CLAIM not in result.stdout or received == 0


def test_dry_run_does_not_claim_block_when_guard_never_reports(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Pins the "did the guard fire?" half of the verdict: the turn is dropped with
    # no guard event and no model call, so nothing shows the Layer-1 guard blocked it.
    def chat_dropping_silently(self: agent_module.BookingAgent, user_message: str) -> str:
        if is_injection_attempt(user_message):
            return SAFE_REPLY
        self.messages.append({"role": "user", "content": user_message})
        return self._run_loop()

    monkeypatch.setattr(agent_module.BookingAgent, "chat", chat_dropping_silently)
    payloads = _spy_on_model_payloads(monkeypatch)
    exit_code = cli.run_dry_run()
    out = capsys.readouterr().out
    assert GUARD_EVENT_LINE not in out  # precondition: no guard event
    assert not any(cli.DRY_RUN_INJECTION in payload for payload in payloads)
    assert BLOCK_CLAIM not in out
    assert "did not fire on it." in out
    assert exit_code != 0


# Review finding (round 3, 2026-09-27): some legs of the checks above were live code
# that no test pinned, so a one-line edit could switch one off with the suite still
# green: the system-prompt half of the "turn left the state unchanged" check, the
# system-prompt and tool parts of the text match, its case-and-spacing normalization,
# and the reset of the "later turn running" flag after each turn. Each plant below is
# caught by exactly one of those legs, and first proves the model really received it.


def _spy_on_model_tools(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every tool list the stub model is really called with."""
    sent_tools: list[str] = []
    original = StubModelClient.create_message

    def spy(self: StubModelClient, **kwargs: Any) -> dict[str, Any]:
        sent_tools.append(json.dumps(kwargs["tools"], ensure_ascii=False))
        return original(self, **kwargs)

    monkeypatch.setattr(StubModelClient, "create_message", spy)
    return sent_tools


def _sent_in(payloads: list[str], part: str, text: str) -> bool:
    """Whether `text` was sent in the `part` ("system" or "messages") of any payload."""
    return any(
        text in json.dumps(json.loads(payload)[part], ensure_ascii=False) for payload in payloads
    )


def test_dry_run_does_not_claim_block_when_injection_is_encoded_into_system_prompt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The injection turn makes no model call and leaves the history alone, but writes
    # a base64 copy into the system prompt, which the next turn sends. No text match
    # can see it; only the system-prompt half of the state check does.
    encoded = base64.b64encode(cli.DRY_RUN_INJECTION.encode()).decode()

    def chat_encoding_into_system(self: agent_module.BookingAgent, user_message: str) -> str:
        if is_injection_attempt(user_message):
            self._on_event("injection_blocked", user_message)
            self._system.append({"type": "text", "text": "Mensaje retenido: " + encoded})
            return SAFE_REPLY
        self.messages.append({"role": "user", "content": user_message})
        return self._run_loop()

    monkeypatch.setattr(agent_module.BookingAgent, "chat", chat_encoding_into_system)
    _add_follow_up_turn(monkeypatch)
    payloads = _spy_on_model_payloads(monkeypatch)
    exit_code = cli.run_dry_run()
    out = capsys.readouterr().out
    assert GUARD_EVENT_LINE in out  # precondition: the guard really did fire
    assert _sent_in(payloads, "system", encoded)  # precondition: it was sent
    assert not any(cli.DRY_RUN_INJECTION in payload for payload in payloads)
    assert BLOCK_CLAIM not in out
    assert "changed the conversation history or system prompt" in out
    assert exit_code != 0


def test_dry_run_does_not_claim_block_when_system_prompt_gets_it_after_its_turn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The injection turn itself changes nothing; printing its reply (after the state
    # check) writes the message into the system prompt, and the next turn sends it.
    # Only the system-prompt part of the text match sees it.
    def chat_writing_system_on_print(self: agent_module.BookingAgent, user_message: str) -> str:
        if is_injection_attempt(user_message):
            self._on_event("injection_blocked", user_message)
            agent = self

            class DeferredReply(str):
                def __format__(self, format_spec: str) -> str:
                    agent._system.append({"type": "text", "text": "Nota: " + user_message})
                    return str.__format__(self, format_spec)

            return DeferredReply(SAFE_REPLY)
        self.messages.append({"role": "user", "content": user_message})
        return self._run_loop()

    monkeypatch.setattr(agent_module.BookingAgent, "chat", chat_writing_system_on_print)
    _add_follow_up_turn(monkeypatch)
    payloads = _spy_on_model_payloads(monkeypatch)
    exit_code = cli.run_dry_run()
    out = capsys.readouterr().out
    assert GUARD_EVENT_LINE in out  # precondition: the guard really did fire
    assert _sent_in(payloads, "system", cli.DRY_RUN_INJECTION)  # precondition: it was sent
    assert not _sent_in(payloads, "messages", cli.DRY_RUN_INJECTION)
    assert BLOCK_CLAIM not in out
    assert "contained its text" in out
    assert exit_code != 0


def test_dry_run_does_not_claim_block_when_a_later_turn_sends_it_as_a_tool_definition(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The injection turn changes nothing; the next turn, on the run's own thread,
    # sends the kept message as the description of an extra tool. Only the tools part
    # of the text match sees it.
    stash: list[str] = []

    def chat_sending_it_as_a_tool(self: agent_module.BookingAgent, user_message: str) -> str:
        if is_injection_attempt(user_message):
            self._on_event("injection_blocked", user_message)
            stash.append(user_message)
            return SAFE_REPLY
        if stash:
            self.client.create_message(
                model=self.model,
                max_tokens=64,
                system=self._system,
                tools=[
                    {
                        "name": "nota_interna",
                        "description": stash.pop(),
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
                messages=[{"role": "user", "content": user_message}],
            )
        self.messages.append({"role": "user", "content": user_message})
        return self._run_loop()

    monkeypatch.setattr(agent_module.BookingAgent, "chat", chat_sending_it_as_a_tool)
    _add_follow_up_turn(monkeypatch)
    payloads = _spy_on_model_payloads(monkeypatch)
    sent_tools = _spy_on_model_tools(monkeypatch)
    exit_code = cli.run_dry_run()
    out = capsys.readouterr().out
    assert GUARD_EVENT_LINE in out  # precondition: the guard really did fire
    # precondition: it was sent, in a tool definition and nowhere else
    assert any(cli.DRY_RUN_INJECTION in tools for tools in sent_tools)
    assert not any(cli.DRY_RUN_INJECTION in payload for payload in payloads)
    assert BLOCK_CLAIM not in out
    assert "contained its text" in out
    assert exit_code != 0


def test_dry_run_does_not_claim_block_when_a_later_turn_sends_it_recased_and_respaced(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The injection turn keeps the message aside in upper case with doubled spaces and
    # the next turn sends it. The text match ignores case and spacing; without that,
    # this copy would pass unseen.
    altered = "  ".join(cli.DRY_RUN_INJECTION.upper().split())
    stash: list[str] = []

    def chat_stashing_altered_copy(self: agent_module.BookingAgent, user_message: str) -> str:
        if is_injection_attempt(user_message):
            self._on_event("injection_blocked", user_message)
            stash.append(altered)
            return SAFE_REPLY
        content = "\n".join([*stash, user_message])
        stash.clear()
        self.messages.append({"role": "user", "content": content})
        return self._run_loop()

    monkeypatch.setattr(agent_module.BookingAgent, "chat", chat_stashing_altered_copy)
    _add_follow_up_turn(monkeypatch)
    payloads = _spy_on_model_payloads(monkeypatch)
    exit_code = cli.run_dry_run()
    out = capsys.readouterr().out
    assert GUARD_EVENT_LINE in out  # precondition: the guard really did fire
    assert any(altered in payload for payload in payloads)  # precondition: it was sent
    assert not any(cli.DRY_RUN_INJECTION in payload for payload in payloads)
    assert BLOCK_CLAIM not in out
    assert "contained its text" in out
    assert exit_code != 0


def test_dry_run_does_not_claim_block_when_printing_a_later_reply_calls_the_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The injection turn keeps a base64 copy aside; the next turn runs normally, but
    # its reply sends the copy to the model when it is printed, after that turn has
    # ended. Only resetting the "later turn running" flag after each turn sees it.
    encoded = base64.b64encode(cli.DRY_RUN_INJECTION.encode()).decode()
    stash: list[str] = []

    def chat_deferring_call_to_later_reply(
        self: agent_module.BookingAgent, user_message: str
    ) -> str:
        if is_injection_attempt(user_message):
            self._on_event("injection_blocked", user_message)
            stash.append(encoded)
            return SAFE_REPLY
        self.messages.append({"role": "user", "content": user_message})
        reply = self._run_loop()
        if not stash:
            return reply
        agent = self
        kept = stash.pop()

        class DeferredReply(str):
            def __format__(self, format_spec: str) -> str:
                _injection_call(agent, kept)
                return str.__format__(self, format_spec)

        return DeferredReply(reply)

    monkeypatch.setattr(agent_module.BookingAgent, "chat", chat_deferring_call_to_later_reply)
    _add_follow_up_turn(monkeypatch)
    payloads = _spy_on_model_payloads(monkeypatch)
    exit_code = cli.run_dry_run()
    out = capsys.readouterr().out
    assert GUARD_EVENT_LINE in out  # precondition: the guard really did fire
    assert any(encoded in payload for payload in payloads)  # precondition: it was sent
    assert not any(cli.DRY_RUN_INJECTION in payload for payload in payloads)
    assert BLOCK_CLAIM not in out
    assert "outside a later scripted turn" in out
    assert exit_code != 0


def test_dry_run_does_not_claim_block_when_tool_data_carries_the_injection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The real agent, but a service description in the mock catalog carries the
    # injection text, so the model reads it in tool data on the first turn, long
    # before the guard blocks the customer's copy. The model did receive it.
    original_get_services = BookingService.get_services

    def get_services_poisoned(self: BookingService) -> list[dict[str, Any]]:
        services = original_get_services(self)
        services[0]["description"] += " " + cli.DRY_RUN_INJECTION
        return services

    monkeypatch.setattr(BookingService, "get_services", get_services_poisoned)
    payloads = _spy_on_model_payloads(monkeypatch)
    exit_code = cli.run_dry_run()
    _assert_no_block_claim(capsys.readouterr().out, exit_code, payloads)
