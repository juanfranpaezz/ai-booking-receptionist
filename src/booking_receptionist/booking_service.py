"""In-memory mock booking domain — no database, no commercial logic.

This is the *fake* business the receptionist agent serves: a fictional Pilates
studio with a handful of services and pre-seeded availability. Every method
returns plain Python data structures so the tool layer can serialize them to
JSON for the model.

In the production system this role is filled by a real database (services,
slots, appointments, multi-tenant company data). Here it is a deterministic,
self-contained stub so the demo runs anywhere with zero setup.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta


@dataclass(frozen=True)
class Service:
    service_id: int
    name: str
    duration_min: int
    price_ars: int
    description: str


@dataclass(frozen=True)
class Slot:
    slot_id: int
    service_id: int
    start: datetime
    professional: str


@dataclass
class Appointment:
    appointment_id: int
    slot_id: int
    service_id: int
    client_name: str
    client_phone: str
    client_email: str | None
    start: datetime
    professional: str
    status: str = "confirmed"  # "confirmed" | "cancelled"


# A fixed "today" makes the demo and the recorded transcript reproducible.
DEMO_TODAY = date(2026, 6, 22)  # a Monday


class BookingService:
    """A fictional Pilates studio with in-memory services, slots, appointments.

    Deterministic by construction: the same seed produces the same catalog and
    the same availability every run, so the transcript in the README is exactly
    what a fresh run reproduces in --dry-run mode.
    """

    COMPANY = {
        "name": "Estudio Pilates Demo",
        "description": (
            "Estudio boutique de Pilates reformer y mat en Palermo, Buenos Aires. "
            "Clases reducidas (máx. 6 personas) y sesiones individuales."
        ),
        "phone": "+54 11 5555-0123",
        "email": "hola@pilatesdemo.example",
        "address": "Av. Demo 1234, Palermo, CABA",
        "hours": "Lunes a viernes 8:00–20:00, sábados 9:00–13:00",
    }

    def __init__(self, today: date = DEMO_TODAY) -> None:
        self.today = today
        self._services: dict[int, Service] = {}
        self._slots: dict[int, Slot] = {}
        self._appointments: dict[int, Appointment] = {}
        self._appt_counter = itertools.count(start=9001)
        self._seed()

    # ---- seeding -------------------------------------------------------

    def _seed(self) -> None:
        services = [
            Service(1, "Pilates Reformer (individual)", 55, 18000,
                    "Sesión 1 a 1 en reformer con instructor certificado."),
            Service(2, "Pilates Reformer (grupal)", 55, 9500,
                    "Clase grupal reducida (máx. 6) en reformer."),
            Service(3, "Pilates Mat (grupal)", 50, 7000,
                    "Clase grupal de mat con elementos (banda, pelota, aro)."),
            Service(4, "Evaluación postural inicial", 40, 12000,
                    "Primera consulta: evaluación y plan personalizado."),
        ]
        for s in services:
            self._services[s.service_id] = s

        # Pre-seed slots for the next 7 days, a few per service per day.
        # Slot ids are assigned sequentially from 101 in a deterministic order.
        slot_id = itertools.count(start=101)
        slot_times = {
            1: [time(9, 0), time(15, 0)],
            2: [time(8, 0), time(18, 0)],
            3: [time(10, 0), time(19, 0)],
            4: [time(11, 0)],
        }
        professionals = {1: "Lucía", 2: "Mateo", 3: "Lucía", 4: "Mateo"}
        for day_offset in range(0, 7):
            d = self.today + timedelta(days=day_offset)
            if d.weekday() == 6:  # studio closed Sundays
                continue
            for service_id, times in slot_times.items():
                for t in times:
                    sid = next(slot_id)
                    self._slots[sid] = Slot(
                        slot_id=sid,
                        service_id=service_id,
                        start=datetime.combine(d, t),
                        professional=professionals[service_id],
                    )

    # ---- read operations (called by tools) -----------------------------

    def get_company_info(self) -> dict:
        return dict(self.COMPANY)

    def get_services(self) -> list[dict]:
        return [
            {
                "service_id": s.service_id,
                "name": s.name,
                "duration_min": s.duration_min,
                "price_ars": s.price_ars,
                "description": s.description,
            }
            for s in self._services.values()
        ]

    def get_available_slots(
        self,
        service_id: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict]:
        """Free slots, optionally filtered by service and date window.

        Dates are ISO strings (YYYY-MM-DD). With no window, returns the next
        7 days. Slots already booked by a confirmed appointment are excluded.
        """
        lo = _parse_date(start_date) or self.today
        hi = _parse_date(end_date) or (self.today + timedelta(days=7))
        booked = {
            a.slot_id for a in self._appointments.values() if a.status == "confirmed"
        }
        out = []
        for slot in self._slots.values():
            if slot.slot_id in booked:
                continue
            if service_id is not None and slot.service_id != service_id:
                continue
            if not (lo <= slot.start.date() <= hi):
                continue
            out.append(
                {
                    "slot_id": slot.slot_id,
                    "service_id": slot.service_id,
                    "service_name": self._services[slot.service_id].name,
                    "start": slot.start.isoformat(timespec="minutes"),
                    "professional": slot.professional,
                }
            )
        out.sort(key=lambda r: r["start"])
        return out

    def get_client_appointments(self, client_name: str, client_phone: str) -> list[dict]:
        """Appointments owned by THIS caller only.

        Ownership is keyed on phone — the caller cannot read another client's
        bookings even if they pass a different name. In the production system
        the session phone overrides any phone the model supplies; here we take
        the phone as the trusted ownership key directly.
        """
        return [
            self._appt_dict(a)
            for a in self._appointments.values()
            if a.client_phone == client_phone
        ]

    # ---- write operations (called by tools) ----------------------------

    def create_appointment(
        self,
        slot_id: int,
        client_name: str,
        client_phone: str,
        client_email: str | None = None,
    ) -> dict:
        slot = self._slots.get(slot_id)
        if slot is None:
            return {"error": "slot_not_found", "message": f"No existe el turno {slot_id}."}
        already = any(
            a.slot_id == slot_id and a.status == "confirmed"
            for a in self._appointments.values()
        )
        if already:
            return {"error": "slot_taken", "message": "Ese turno ya fue reservado."}
        appt = Appointment(
            appointment_id=next(self._appt_counter),
            slot_id=slot_id,
            service_id=slot.service_id,
            client_name=client_name,
            client_phone=client_phone,
            client_email=client_email,
            start=slot.start,
            professional=slot.professional,
        )
        self._appointments[appt.appointment_id] = appt
        return self._appt_dict(appt)

    def cancel_appointment(self, appointment_id: int, client_phone: str) -> dict:
        appt = self._appointments.get(appointment_id)
        if appt is None:
            return {"error": "not_found", "message": f"No existe el turno {appointment_id}."}
        if appt.client_phone != client_phone:
            # Ownership gate: cannot cancel someone else's appointment.
            return {"error": "forbidden", "message": "Ese turno no pertenece a este número."}
        appt.status = "cancelled"
        return {"appointment_id": appointment_id, "status": "cancelled"}

    # ---- helpers -------------------------------------------------------

    def _appt_dict(self, a: Appointment) -> dict:
        return {
            "appointment_id": a.appointment_id,
            "service_id": a.service_id,
            "service_name": self._services[a.service_id].name,
            "client_name": a.client_name,
            "start": a.start.isoformat(timespec="minutes"),
            "professional": a.professional,
            "status": a.status,
        }


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None
