"""Log-normal duration generation for realistic event timing.

LLM calls and tool invocations follow log-normal distributions in practice:
most calls are fast, but there's a long tail of slow ones. This module
provides a seeded generator for reproducible timestamp sequences.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta


class TimestampGenerator:
    """Generates realistic timestamps with log-normal inter-event durations."""

    def __init__(self, rng: random.Random, base_time: datetime | None = None) -> None:
        self._rng = rng
        self._current = base_time or datetime(2025, 1, 15, 9, 0, 0, tzinfo=UTC)

    @property
    def current(self) -> datetime:
        return self._current

    def advance(self, mean_ms: float = 500.0, sigma: float = 0.8) -> datetime:
        """Advance time by a log-normal duration and return the new timestamp.

        Args:
            mean_ms: Mean duration in milliseconds.
            sigma: Standard deviation of the underlying normal distribution.
                   Higher values → more variance (longer tail).
        """
        # Log-normal: if X ~ N(mu, sigma^2), then e^X ~ LogNormal
        # We want E[e^X] = mean_ms, so mu = ln(mean_ms) - sigma^2/2
        mu = math.log(mean_ms) - (sigma ** 2) / 2.0
        duration_ms = self._rng.lognormvariate(mu, sigma)
        self._current += timedelta(milliseconds=duration_ms)
        return self._current

    def advance_fixed(self, ms: float) -> datetime:
        """Advance by an exact duration (for deterministic gaps)."""
        self._current += timedelta(milliseconds=ms)
        return self._current
