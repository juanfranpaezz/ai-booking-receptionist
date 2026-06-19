"""Tool (function) definitions the Claude model is allowed to call.

These are the structured-I/O contract between the model and the booking domain.
Each entry is a JSON-Schema tool definition per the Anthropic Messages API; the
agent loop dispatches a `tool_use` block to `execute_tool`, which calls the
mock BookingService and returns JSON the model reads back as a `tool_result`.

Design notes carried over from the production system:
  * Tool allow-listing IS the security boundary. The model can ONLY do what a
    tool exposes. An admin-only `get_stats` capability exists in the dispatcher
    but is deliberately NOT in TOOLS, so the model can never call it.
  * Ownership is enforced at the tool-execution layer, not in the prompt: the
    caller's phone is injected from the trusted session, overriding anything the
    model passes. The model cannot read or cancel another client's appointments
    even if steered to try.
  * The LAST tool carries `cache_control` so the whole tool block + system prompt
    cache as one stable prefix (see agent.py / README → Prompt caching).
"""

from __future__ import annotations

import json

from .booking_service import BookingService

# Ordered list — order is stable so the cached prefix stays byte-identical.
TOOLS: list[dict] = [
    {
        "name": "get_company_info",
        "description": (
            "Devuelve la información del estudio: nombre, descripción, teléfono, "
            "email, dirección y horarios. Usala cuando el cliente pregunte datos "
            "de contacto o ubicación."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_services",
        "description": (
            "Lista los servicios activos con su nombre, duración, precio y "
            "descripción. Usala cuando el cliente pregunte qué clases hay o cuánto "
            "cuestan, y SIEMPRE antes de reservar para conocer el service_id."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_available_slots",
        "description": (
            "Busca turnos disponibles. Si no se especifica fecha, busca los "
            "próximos 7 días. Devuelve slot_id, servicio, fecha/hora y profesional. "
            "Usá el slot_id exacto al reservar."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "service_id": {
                    "type": "integer",
                    "description": "ID del servicio (opcional, para filtrar).",
                },
                "start_date": {
                    "type": "string",
                    "description": "Fecha inicial YYYY-MM-DD (opcional).",
                },
                "end_date": {
                    "type": "string",
                    "description": "Fecha final YYYY-MM-DD (opcional).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_client_appointments",
        "description": (
            "Devuelve ÚNICAMENTE los turnos del cliente que está hablando, nunca "
            "los de otra persona. Usala antes de cancelar para mostrarle sus turnos."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "client_name": {
                    "type": "string",
                    "description": "Nombre del cliente actual.",
                },
            },
            "required": ["client_name"],
        },
    },
    {
        "name": "cancel_appointment",
        "description": (
            "Cancela un turno por su ID. Solo después de buscar los turnos del "
            "cliente, mostrárselos y obtener su confirmación explícita."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "appointment_id": {
                    "type": "integer",
                    "description": "ID del turno a cancelar.",
                },
            },
            "required": ["appointment_id"],
        },
    },
    {
        "name": "create_appointment",
        "description": (
            "Reserva un turno. Solo después de confirmar el servicio, mostrar la "
            "disponibilidad, que el cliente elija un slot_id, y recolectar nombre "
            "(email opcional) con confirmación explícita."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "slot_id": {
                    "type": "integer",
                    "description": "ID del turno disponible elegido.",
                },
                "client_name": {"type": "string", "description": "Nombre del cliente."},
                "client_email": {
                    "type": "string",
                    "description": "Email del cliente (opcional).",
                },
            },
            "required": ["slot_id", "client_name"],
        },
        # Last tool carries the cache breakpoint: tools + system cache as one prefix.
        "cache_control": {"type": "ephemeral"},
    },
]

# Names the model is allowed to call (derived from TOOLS — single source of truth).
ALLOWED_TOOL_NAMES = {t["name"] for t in TOOLS}


def execute_tool(
    name: str,
    tool_input: dict,
    service: BookingService,
    *,
    session_phone: str,
) -> str:
    """Run one tool against the mock domain and return a JSON string.

    `session_phone` is the trusted caller identity from the session. It OVERRIDES
    any phone the model might pass — the model never controls whose data it reads
    or mutates. Returns JSON (the model reads it back as tool_result content).
    """
    try:
        if name not in ALLOWED_TOOL_NAMES:
            # Defense in depth: the loop should never dispatch an unknown tool,
            # but if it somehow does, refuse rather than guess.
            return json.dumps({"error": "tool_not_allowed", "tool": name})

        if name == "get_company_info":
            result = service.get_company_info()
        elif name == "get_services":
            result = service.get_services()
        elif name == "get_available_slots":
            result = service.get_available_slots(
                service_id=tool_input.get("service_id"),
                start_date=tool_input.get("start_date"),
                end_date=tool_input.get("end_date"),
            )
        elif name == "get_client_appointments":
            # client_phone comes from the session, NOT from the model.
            result = service.get_client_appointments(
                client_name=tool_input.get("client_name", ""),
                client_phone=session_phone,
            )
        elif name == "cancel_appointment":
            result = service.cancel_appointment(
                appointment_id=int(tool_input["appointment_id"]),
                client_phone=session_phone,
            )
        elif name == "create_appointment":
            result = service.create_appointment(
                slot_id=int(tool_input["slot_id"]),
                client_name=tool_input.get("client_name", ""),
                client_phone=session_phone,
                client_email=tool_input.get("client_email"),
            )
        else:  # pragma: no cover - unreachable given the allow-list check above
            result = {"error": "unknown_tool", "tool": name}

        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as exc:  # noqa: BLE001 - never leak a stack trace to the model/user
        # Generic error: no internal details, table names, or stack traces escape.
        return json.dumps(
            {"error": "tool_execution_failed", "message": "No se pudo completar la acción."}
        )
