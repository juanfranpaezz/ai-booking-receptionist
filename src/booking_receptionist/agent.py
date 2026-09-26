"""The Claude tool-use agentic loop — the interesting engineering.

This is the heart of the showcase: a manual agentic loop over the Anthropic
Messages API that lets the model book appointment slots by calling tools.

Control flow per user turn:
    1. Layer-1 injection pre-filter — if the incoming message is a known
       override/jailbreak attempt, return a canned safe reply and NEVER touch
       the model (cheapest possible block).
    2. Otherwise, run the agentic loop (bounded at MAX_ITERATIONS):
         a. Call the model with the cached system prompt + cached tool block.
         b. stop_reason == "end_turn"  -> return the text (done).
         c. stop_reason == "tool_use"  -> execute each requested tool against the
            mock domain (phone forced from the trusted session), append the
            assistant turn + a user turn carrying the tool_result(s), loop.
         d. anything else / loop exhausted -> safe fallback message.

The loop is written manually (rather than the SDK's `tool_runner`) on purpose:
it mirrors the production design and shows the gate points — per-tool ownership
enforcement, the iteration bound, and the prompt-cache breakpoints.

`--dry-run` mode swaps in a deterministic STUB model (no network, no API key) so
the whole loop — including a real tool call and result round-trip — is exercised
end to end. That is what `python -m booking_receptionist --dry-run` runs.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any, Protocol, cast

from .booking_service import BookingService
from .injection_guard import SAFE_REPLY, is_injection_attempt
from .system_prompt import build_system_prompt
from .tools import TOOLS, execute_tool

# Default to Sonnet: cheaper than Opus, plenty capable for a structured
# tool-use receptionist, and the sensible pick for a low-cost public demo.
# (The production system routed WhatsApp traffic to Haiku for cost; this demo
# keeps it simple with one model.)
DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048
MAX_ITERATIONS = 5  # runaway guard: bound the tool-call loop per user turn.


class ModelClient(Protocol):
    """Minimal interface the agent needs from a model backend.

    Both the real Anthropic SDK wrapper and the dry-run stub satisfy this, so the
    loop is identical in both modes.
    """

    def create_message(
        self,
        *,
        model: str,
        max_tokens: int,
        system: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return a response dict with keys: content (list of blocks), stop_reason, usage."""
        ...


class BookingAgent:
    """Stateful per-conversation agent over a mock BookingService."""

    def __init__(
        self,
        service: BookingService | None = None,
        model_client: ModelClient | None = None,
        *,
        session_phone: str = "+5491100000000",
        model: str = DEFAULT_MODEL,
        on_event: Callable[[str, Any], None] | None = None,
    ) -> None:
        self.service = service or BookingService()
        self.model = model
        self.session_phone = session_phone
        self.messages: list[dict[str, Any]] = []
        self._on_event = on_event or (lambda *_: None)
        # System prompt is built once and cached (stable prefix for prompt caching).
        self._system = [
            {
                "type": "text",
                "text": build_system_prompt(self.service),
                "cache_control": {"type": "ephemeral"},
            }
        ]
        self.client = model_client or _build_default_client(model)

    # ---- public API ----------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Process one user turn and return the assistant's reply text."""
        # Layer 1: block known injections before spending a single token.
        if is_injection_attempt(user_message):
            self._on_event("injection_blocked", user_message)
            return SAFE_REPLY

        self.messages.append({"role": "user", "content": user_message})
        return self._run_loop()

    # ---- the agentic loop ----------------------------------------------

    def _run_loop(self) -> str:
        for _iteration in range(MAX_ITERATIONS):
            response = self.client.create_message(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=self._system,
                tools=TOOLS,
                messages=self.messages,
            )
            self._record_cache_usage(response.get("usage", {}))
            stop_reason = response.get("stop_reason")
            content = response.get("content", [])

            if stop_reason == "end_turn":
                # Append the assistant turn and return its first text block.
                self.messages.append({"role": "assistant", "content": content})
                return _first_text(content)

            if stop_reason == "tool_use":
                # Append the assistant turn (with the tool_use blocks) verbatim.
                self.messages.append({"role": "assistant", "content": content})
                tool_results = []
                for block in content:
                    if block.get("type") != "tool_use":
                        continue
                    name = block["name"]
                    self._on_event("tool_use", {"name": name, "input": block.get("input", {})})
                    result_json = execute_tool(
                        name,
                        block.get("input", {}),
                        self.service,
                        session_phone=self.session_phone,
                    )
                    self._on_event("tool_result", {"name": name, "result": result_json})
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": result_json,
                        }
                    )
                # Feed all tool results back as a single user turn, then loop.
                self.messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop reason (refusal, max_tokens, etc.) — fail safe.
            self._on_event("unexpected_stop", stop_reason)
            return (
                "Disculpá, tuve un inconveniente para procesar tu mensaje. "
                "¿Podés reformularlo?"
            )

        # Loop exhausted without a final answer.
        self._on_event("loop_exhausted", MAX_ITERATIONS)
        return (
            "Disculpá, esto está tomando más de lo esperado. "
            "¿Querés que lo intentemos de nuevo?"
        )

    def _record_cache_usage(self, usage: dict[str, Any]) -> None:
        created = usage.get("cache_creation_input_tokens", 0)
        read = usage.get("cache_read_input_tokens", 0)
        if created or read:
            self._on_event("cache", {"creation": created, "read": read})


# ---- helpers -----------------------------------------------------------

def _first_text(content: list[dict[str, Any]]) -> str:
    for block in content:
        if block.get("type") == "text":
            text = block.get("text", "")
            return text if isinstance(text, str) else ""
    return ""


def _build_default_client(model: str) -> ModelClient:
    """Real Anthropic-SDK-backed client. Requires ANTHROPIC_API_KEY in the env.

    Imported lazily so the package (and `--dry-run`) works with the `anthropic`
    SDK absent or no key set.
    """
    return AnthropicModelClient(model=model)


class AnthropicModelClient:
    """Thin wrapper over the official `anthropic` Python SDK.

    Reads the API key from the ANTHROPIC_API_KEY environment variable BY NAME —
    it is never hardcoded, logged, or accepted as an argument. If the key or SDK
    is missing, construction fails closed with a clear message.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it, or run with --dry-run "
                "to exercise the loop against the stubbed model (no key needed)."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'anthropic' package is not installed. Run "
                "`pip install -e .` (or `pip install anthropic`), or use --dry-run."
            ) from exc
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY itself
        self.model = model

    def create_message(
        self,
        *,
        model: str,
        max_tokens: int,
        system: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # The loop deliberately carries plain dicts (the same shape the REST API
        # takes) rather than the SDK's TypedDicts, so the stub and the live client
        # are interchangeable. The casts say exactly that and nothing more.
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=cast("Any", system),
            tools=cast("Any", tools),
            messages=cast("Any", messages),
        )
        # Normalize the SDK object into the plain dict shape the loop expects.
        return {
            "content": [_block_to_dict(b) for b in response.content],
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cache_creation_input_tokens": getattr(
                    response.usage, "cache_creation_input_tokens", 0
                )
                or 0,
                "cache_read_input_tokens": getattr(
                    response.usage, "cache_read_input_tokens", 0
                )
                or 0,
            },
        }


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Convert an SDK content block into the plain-dict form the loop appends.

    The dict must round-trip back to the API as an assistant content block, so we
    preserve type/text for text blocks and id/name/input for tool_use blocks.
    """
    btype = getattr(block, "type", None)
    if btype == "text":
        return {"type": "text", "text": block.text}
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    # Fallback: best-effort serialization for any other block type.
    if hasattr(block, "model_dump"):
        return cast("dict[str, Any]", block.model_dump())
    return cast("dict[str, Any]", json.loads(json.dumps(block, default=str)))
