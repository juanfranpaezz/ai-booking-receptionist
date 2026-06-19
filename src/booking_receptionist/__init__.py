"""AI Booking Receptionist — a Claude tool-use agent demo.

A sanitized, self-contained extract of the production pattern behind a real
WhatsApp AI receptionist: a Claude tool-use agentic loop that books appointment
slots, with prompt-injection defense and prompt caching.

The domain is a mock in-memory studio ("Estudio Pilates Demo") — no database,
no commercial logic. See README.md.
"""

__version__ = "0.1.0"
