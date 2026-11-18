"""Deterministic synthetic conversations and turn-indexed recall facts.

Facts record what was asserted at each turn, including deliberately conflicting
assertions. They are not a resolved view of the project's current requirements.
"""

from dataclasses import dataclass


PROFILES = ('growing', 'pivot', 'contradiction')
LENGTHS = (1, 3, 6, 10, 20)

# Each entry pairs a message with a literal, memory-friendly recall snippet.
# Overrides introduce the pivot and budget contradictions at their named turns.
_GROWING_FACTS = (
    ('The project_name is Harbor. Build a React web portal for scheduling appointments.', 'Harbor'),
    ('Add customer accounts.', 'customer accounts'),
    ('Add a calendar for appointments.', 'calendar for appointments'),
    ('Allow customers to cancel appointments.', 'cancel appointments'),
    ('Send appointment reminders.', 'appointment reminders'),
    ('Use cloud hosting.', 'cloud hosting'),
    ('Allow staff to set availability.', 'staff to set availability'),
    ('Keep an audit log of appointment changes.', 'audit log of appointment changes'),
    ('Support recurring appointments.', 'recurring appointments'),
    ('Allow staff to block holidays.', 'block holidays'),
    ('Send a confirmation after booking.', 'confirmation after booking'),
    ('Show available time slots.', 'available time slots'),
    ('Allow staff to add appointment notes.', 'appointment notes'),
    ('Provide a daily appointment list.', 'daily appointment list'),
    ('Allow customers to reschedule.', 'customers to reschedule'),
    ('Record the appointment duration.', 'appointment duration'),
    ('Allow staff to search appointments.', 'search appointments'),
    ('Show a cancellation confirmation.', 'cancellation confirmation'),
    ('Allow staff to export the schedule.', 'export the schedule'),
    ('Provide an accessibility review for booking.', 'accessibility review for booking'),
)
_OVERRIDES = {
    'pivot': {
        1: (_GROWING_FACTS[0][0], 'React'),
        5: ('We pivot the technology stack from React to Flutter.', 'Flutter'),
    },
    'contradiction': {
        3: ('El requisito es presupuesto 30k€ para todo el proyecto.', 'presupuesto 30k€'),
        8: ('El requisito es presupuesto 80k€ para todo el proyecto.', 'presupuesto 80k€'),
    },
}


@dataclass(frozen=True)
class Turn:
    number: int
    message: str


@dataclass(frozen=True)
class Scenario:
    profile: str
    turns: tuple[Turn, ...]
    fact_tracker: dict[int, str]


def generate_scenario(profile: str, length: int) -> Scenario:
    """Return a fresh conversation and exact recall phrase for each turn.

    Turn numbers and fact-tracker keys are one-based. Each fact is a literal
    snippet in its message, without a turn label, for case-insensitive matching.
    """
    if profile not in PROFILES:
        raise ValueError(f'Unknown scenario profile: {profile!r}')
    if type(length) is not int or length not in LENGTHS:
        raise ValueError(f'Unsupported scenario length: {length!r}')

    overrides = _OVERRIDES.get(profile, {})
    entries = [overrides.get(number, _GROWING_FACTS[number - 1])
               for number in range(1, length + 1)]
    turns = tuple(Turn(number, f'Synthetic scheduling project. {message}')
                  for number, (message, _) in enumerate(entries, start=1))
    facts = {number: fact for number, (_, fact) in enumerate(entries, start=1)}
    return Scenario(profile=profile, turns=turns, fact_tracker=facts)
