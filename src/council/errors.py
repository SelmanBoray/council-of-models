"""Exception hierarchy for council-of-models."""

from __future__ import annotations


class CouncilError(RuntimeError):
    """Base class for all council-of-models errors."""


class ProviderError(CouncilError):
    """A model provider failed: missing API key, HTTP error, refusal, empty output..."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
