"""Prompt-injection defense.

A sanitized reproduction of the production system's multi-layer defense. Two of
the four production layers are interesting enough to show standalone here:

  Layer 1 — runtime pre-filter (`is_injection_attempt`):
      Scans the *incoming user message* for known jailbreak / instruction-override
      patterns. On a match the agent returns a canned safe reply and NEVER calls
      the model — the cheapest possible block (no tokens spent, no chance for the
      model to be steered).

  Layer 2 — stored-payload sanitizer (`sanitize_for_prompt`):
      Applied to every piece of *business data* (service names, descriptions,
      company fields) before it is interpolated into the system prompt. Defends
      against a second-order injection where an attacker persists malicious text
      in a data field that later lands verbatim in the prompt.

The two production layers NOT reproduced here are operational, not architectural:
per-session/per-company rate limiting and a monthly cost cap. They are noted in
the README as deliberately omitted.

Anchored to the real patterns in VINDA's `LLMInputSanitizer.java`; expressed in
Python with the same intent. The patterns are illustrative, not exhaustive — a
real deployment tunes them continuously and treats them as defense-in-depth, NOT
as the sole guard (the immutable system-prompt rules and tool allow-listing are
the other layers; see system_prompt.py and agent.py).
"""

from __future__ import annotations

import re

# Layer 1: incoming-message jailbreak / override patterns (case-insensitive).
# Spanish + English variants — the production receptionist serves Argentine users.
_INJECTION_PATTERNS = [
    r"ignor[aá]\s+.*instruccion",          # "ignora las instrucciones"
    r"olvid[aá]\s+.*instruccion",          # "olvida tus instrucciones"
    r"ignore\s+.*(previous|above|prior)",  # "ignore previous instructions"
    r"forget\s+.*(previous|above|instructions)",
    r"disregard\s+.*(previous|above|instructions)",
    r"new\s+(system\s+)?prompt",
    r"nuevo\s+prompt",
    r"eres\s+ahora",                       # "eres ahora un ..."
    r"you\s+are\s+now\s+",
    r"act\s+as\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"roleplay\s+as",
    r"dan\s+mode",
    r"developer\s+mode",
    r"\[INST\]",
    r"<\|.*\|>",                           # chat-template token smuggling
    r"^\s*system\s*:",                     # leading "System:" role spoof
    r"\n\s*(system|assistant)\s*:",        # mid-message role spoof
    r"###\s*(instruc|system|nueva|override)",
]

_COMPILED = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in _INJECTION_PATTERNS]

# Canned reply when an injection is detected (Layer 1 short-circuits the API call).
SAFE_REPLY = (
    "¡Hola! Soy el asistente del estudio y solo puedo ayudarte con turnos, "
    "servicios y consultas del estudio. ¿Querés reservar, consultar o cancelar un turno?"
)

# Layer 2: markers that, if found inside stored business data, get neutralized
# rather than passed through into the system prompt. Each is prefixed with a
# negative lookbehind for the fence opener so that re-sanitizing already-fenced
# text is a no-op (keeps `sanitize_for_prompt` idempotent).
_FENCE_GUARD = r"(?<!\[DATO-USUARIO: )"
_STORED_MARKERS = [
    r"system\s*:",
    r"\[system",
    r"\[instruc",
    r"###\s*(nuevas?\s+)?instruc",
    r"#\s*override",
    r"\[override\]",
    r"ignor[aá]\s+.*instruccion",
    r"olvid[aá]\s+.*instruccion",
    r"</?(system|instruction|prompt)>",
]
_STORED_COMPILED = [
    re.compile(_FENCE_GUARD + p, re.IGNORECASE | re.UNICODE) for p in _STORED_MARKERS
]

# Zero-width / bidirectional unicode used to smuggle hidden instructions.
_ZERO_WIDTH = re.compile(
    "[​‌‍‎‏‪‫‬‭‮﻿]"
)
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MD_HEADER = re.compile(r"(?m)^#{1,6}\s")


def is_injection_attempt(message: str) -> bool:
    """True if the incoming user message matches a known override/jailbreak pattern.

    The agent uses this BEFORE building any request — a positive match means we
    skip the model entirely and return SAFE_REPLY.
    """
    if not message:
        return False
    return any(pat.search(message) for pat in _COMPILED)


def sanitize_for_prompt(value: str | None, max_len: int = 1000) -> str:
    """Neutralize a stored business-data string before it enters the system prompt.

    Steps (idempotent):
      1. Strip zero-width / bidi unicode (hidden-instruction smuggling).
      2. Collapse runs of 3+ newlines (fake turn boundaries).
      3. Wrap any injection marker in a visible [DATO-USUARIO: ...] fence so the
         model sees it as inert data, not an instruction. This runs BEFORE the
         markdown-header strip so a `### NUEVAS INSTRUCCIONES` marker is fenced
         as a whole unit rather than silently de-hashed into bare text.
      4. Strip any remaining markdown headers (fake section markers).
      5. Truncate and trim.
    """
    if value is None:
        return ""
    text = _ZERO_WIDTH.sub("", value)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    for pat in _STORED_COMPILED:
        text = pat.sub(lambda m: f"[DATO-USUARIO: {m.group(0)}]", text)
    text = _MD_HEADER.sub("", text)
    return text[:max_len].strip()
