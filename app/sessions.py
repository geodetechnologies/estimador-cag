"""Volatile, isolated conversation state for the current server process."""

from dataclasses import dataclass, field
from uuid import uuid4

import anyio
from pydantic import BaseModel, ConfigDict, Field

MAX_TURNS = 6


class ProjectMetadata(BaseModel):
    """Project facts maintained independently of the sliding message window."""

    model_config = ConfigDict(extra='forbid')

    project_name: str | None = None
    assumed_team_size: int | None = Field(default=None, ge=1)
    mentioned_technologies: list[str] = Field(default_factory=list)
    agreed_scope: list[str] = Field(default_factory=list)


class ConversationHistory:
    def __init__(self, max_turns: int = MAX_TURNS):
        if max_turns < 1:
            raise ValueError('max_turns must be positive')
        self.max_turns = max_turns
        self._turns: list[tuple[str, str]] = []

    def add_turn(self, user: str, assistant: str) -> None:
        self._turns.append((user, assistant))
        self._turns = self._turns[-self.max_turns:]

    def to_messages_list(self, system: str) -> list[dict[str, str]]:
        messages = [{'role': 'system', 'content': system}]
        for user, assistant in self._turns:
            messages.extend([{'role': 'user', 'content': user},
                             {'role': 'assistant', 'content': assistant}])
        return messages


@dataclass
class Session:
    history: ConversationHistory
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    lock: anyio.Lock = field(default_factory=anyio.Lock)


_sessions: dict[str, Session] = {}


def create_session(max_turns: int) -> str:
    session_id = str(uuid4())
    _sessions[session_id] = Session(history=ConversationHistory(max_turns))
    return session_id


def get_session(session_id: str) -> Session | None:
    return _sessions.get(session_id)
