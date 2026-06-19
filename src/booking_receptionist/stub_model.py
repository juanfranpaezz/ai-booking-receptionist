"""Deterministic stub model for --dry-run (no network, no API key).

It satisfies the same `ModelClient` interface the real SDK wrapper does, so the
agent loop runs *identically* — same tool-call dispatch, same tool_result
round-trip, same stop conditions — without ever contacting the API.

It is intentionally NOT an LLM: it inspects the latest user/tool state and
returns a canned `tool_use` or `end_turn` response. This proves the loop wiring
end to end (the agent really executes a tool against the mock domain and feeds
the result back) without pretending to reproduce model intelligence. It drives a
single scripted booking conversation; off-script inputs get a generic reply.
"""

from __future__ import annotations

import json


class StubModelClient:
    """Scripts one happy-path booking flow deterministically.

    State machine driven by what the conversation already contains:
      user asks to book  -> tool_use get_services
      after services     -> tool_use get_available_slots
      after slots        -> end_turn (offer options, ask for a choice)
      user picks + name   -> tool_use create_appointment
      after create       -> end_turn (confirm the booking)
    """

    def __init__(self) -> None:
        self._toolu = 0
        self._offered_slot_id: int | None = None  # the slot we proposed to the client

    def _next_tool_id(self) -> str:
        self._toolu += 1
        return f"toolu_stub_{self._toolu:03d}"

    def create_message(self, *, model, max_tokens, system, tools, messages, **_) -> dict:
        last = messages[-1]

        # If the last turn was tool results, decide the next step from them.
        if last["role"] == "user" and _is_tool_result_turn(last):
            results = _tool_result_payloads(last)
            if any(r.get("_tool") == "get_services" for r in results):
                # We have services; now look up availability for the grupal reformer.
                return self._tool_use(
                    "get_available_slots", {"service_id": 2, "start_date": "2026-06-23"}
                )
            if any(r.get("_tool") == "get_available_slots" for r in results):
                slots = next(
                    r["_data"] for r in results if r.get("_tool") == "get_available_slots"
                )
                first = slots[0] if slots else None
                if first:
                    # Remember exactly which slot we proposed, so we book THAT one.
                    self._offered_slot_id = first["slot_id"]
                    text = (
                        f"¡Buenísimo! Para Reformer grupal tengo el slot {first['slot_id']} "
                        f"el {first['start'].replace('T', ' a las ')} con {first['professional']}. "
                        f"¿Te lo reservo? Pasame tu nombre así lo confirmo. 😊"
                    )
                else:
                    text = "Por ahora no tengo turnos para esa fecha. ¿Probamos otro día?"
                return _end_turn(text)
            if any(r.get("_tool") == "create_appointment" for r in results):
                appt = next(
                    r["_data"] for r in results if r.get("_tool") == "create_appointment"
                )
                text = (
                    f"¡Listo, {appt['client_name']}! Te reservé {appt['service_name']} "
                    f"el {appt['start'].replace('T', ' a las ')} con {appt['professional']}. "
                    f"Tu número de turno es {appt['appointment_id']}. ¡Nos vemos! 💪"
                )
                return _end_turn(text)
            return _end_turn("Listo. ¿Algo más en lo que te pueda ayudar?")

        # Otherwise it's a user text turn — react to its content.
        text = _last_user_text(messages).lower()
        if "reservar" in text or "turno" in text or "reformer" in text or "clase" in text:
            # Kick off the booking flow by listing services first.
            return self._tool_use("get_services", {})
        if self._offered_slot_id is not None and (
            "me llamo" in text or "soy " in text or "nombre" in text or "perfecto" in text
        ):
            # User gave a name after we offered a slot — book the slot we offered.
            name = _extract_name(_last_user_text(messages))
            return self._tool_use(
                "create_appointment",
                {"slot_id": self._offered_slot_id, "client_name": name},
            )
        return _end_turn(
            "¡Hola! Soy el asistente de Estudio Pilates Demo. "
            "Puedo ayudarte a reservar, consultar o cancelar un turno. ¿Qué necesitás?"
        )

    def _tool_use(self, name: str, tool_input: dict) -> dict:
        return {
            "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": self._next_tool_id(), "name": name, "input": tool_input}
            ],
            "usage": {"input_tokens": 1200, "output_tokens": 40, "cache_read_input_tokens": 0},
        }


def _end_turn(text: str) -> dict:
    return {
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 1200, "output_tokens": 60, "cache_read_input_tokens": 1100},
    }


def _is_tool_result_turn(turn: dict) -> bool:
    content = turn.get("content")
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def _tool_result_payloads(turn: dict) -> list[dict]:
    """Parse each tool_result's JSON content; tag with the tool that produced it.

    We don't carry the tool name on the result block, so we infer it from the
    payload shape — fine for a deterministic stub.
    """
    out = []
    for block in turn["content"]:
        if not (isinstance(block, dict) and block.get("type") == "tool_result"):
            continue
        try:
            data = json.loads(block["content"])
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
        tool = _infer_tool(data)
        out.append({"_tool": tool, "_data": data})
    return out


def _infer_tool(data) -> str:
    if isinstance(data, list):
        if data and "duration_min" in data[0]:
            return "get_services"
        if data and "slot_id" in data[0]:
            return "get_available_slots"
        return "get_client_appointments"
    if isinstance(data, dict):
        if "appointment_id" in data and data.get("status") == "confirmed":
            return "create_appointment"
        if "appointment_id" in data and data.get("status") == "cancelled":
            return "cancel_appointment"
        if "hours" in data:
            return "get_company_info"
    return "unknown"


def _last_user_text(messages: list[dict]) -> str:
    for turn in reversed(messages):
        if turn["role"] == "user" and isinstance(turn.get("content"), str):
            return turn["content"]
    return ""


def _extract_name(text: str) -> str:
    # Naive name pull for the scripted flow ("soy Lucía", "me llamo Lucía").
    for marker in ("me llamo", "soy", "nombre es"):
        idx = text.lower().find(marker)
        if idx != -1:
            tail = text[idx + len(marker):].strip().strip(".,!").split()
            if tail:
                return tail[0].capitalize()
    return text.strip().split()[0].capitalize() if text.strip() else "Cliente"
