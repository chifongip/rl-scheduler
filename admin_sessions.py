from __future__ import annotations

import secrets
import time
from collections.abc import Callable


DEFAULT_ADMIN_SESSION_TIMEOUT_SECONDS = 300
MAX_ADMIN_SESSION_TIMEOUT_SECONDS = 86_400


class AdminSessionStore:
    def __init__(
        self,
        timeout_seconds: int = DEFAULT_ADMIN_SESSION_TIMEOUT_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_sessions: int = 128,
    ):
        self.timeout_seconds = timeout_seconds
        self._clock = clock
        self._max_sessions = max_sessions
        self._sessions: dict[str, float] = {}

    def create(self) -> str:
        self._prune()
        while len(self._sessions) >= self._max_sessions:
            oldest = min(self._sessions, key=lambda token: self._sessions[token])
            self._sessions.pop(oldest, None)
        token = secrets.token_urlsafe(32)
        self._sessions[token] = self._clock() + self.timeout_seconds
        return token

    def validate(self, token: str) -> bool:
        expiry = self._sessions.get(token)
        if expiry is None:
            return False
        if expiry <= self._clock():
            self._sessions.pop(token, None)
            return False
        return True

    def touch(self, token: str) -> bool:
        if not self.validate(token):
            return False
        self._sessions[token] = self._clock() + self.timeout_seconds
        return True

    def revoke(self, token: str) -> bool:
        return self._sessions.pop(token, None) is not None

    def _prune(self) -> None:
        now = self._clock()
        expired = [token for token, expiry in self._sessions.items() if expiry <= now]
        for token in expired:
            self._sessions.pop(token, None)


def parse_admin_session_timeout(value: str | None) -> int:
    if value is None:
        return DEFAULT_ADMIN_SESSION_TIMEOUT_SECONDS
    try:
        timeout = int(value)
    except ValueError:
        return DEFAULT_ADMIN_SESSION_TIMEOUT_SECONDS
    if timeout < 1 or timeout > MAX_ADMIN_SESSION_TIMEOUT_SECONDS:
        return DEFAULT_ADMIN_SESSION_TIMEOUT_SECONDS
    return timeout
