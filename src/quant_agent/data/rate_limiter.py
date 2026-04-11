"""Token-bucket rate limiter for external API calls."""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe token-bucket rate limiter.

    Args:
        max_calls: Maximum number of calls allowed within *period* seconds.
        period: Time window in seconds.
        warning_threshold: Fraction (0-1) at which to log a warning that the
            limit is being approached.  Set to 1 to disable.
    """

    def __init__(
        self,
        max_calls: int = 200,
        period: float = 60.0,
        warning_threshold: float = 0.8,
    ) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        if period <= 0:
            raise ValueError("period must be positive")
        if not (0 < warning_threshold <= 1.0):
            raise ValueError("warning_threshold must be in (0, 1]")

        self._max_calls = max_calls
        self._period = period
        self._warning_threshold = warning_threshold

        # Token-bucket state
        self._tokens: float = float(max_calls)
        self._last_refill: float = time.monotonic()

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def max_calls(self) -> int:
        return self._max_calls

    @property
    def period(self) -> float:
        return self._period

    @property
    def available_tokens(self) -> float:
        """Return current number of available tokens (approximate, lock-free)."""
        return self._tokens

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Replenish tokens based on elapsed time. Must be called under lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * (self._max_calls / self._period)
        self._tokens = min(self._max_calls, self._tokens + new_tokens)
        self._last_refill = now

    def acquire(self, tokens: int = 1) -> float:
        """Consume *tokens* from the bucket.

        Returns the number of seconds the caller should sleep before
        proceeding.  A return value of 0 means no waiting is needed.

        This method does **not** sleep itself so that callers can decide
        how to handle the wait (e.g., log a message first).
        """
        if tokens <= 0:
            return 0.0

        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens

                # Warn if approaching the limit
                usage_ratio = 1.0 - (self._tokens / self._max_calls)
                if usage_ratio >= self._warning_threshold:
                    logger.warning(
                        "Rate limit approaching for %s: %.0f/%d calls used "
                        "(%.0f%%)",
                        id(self),
                        self._max_calls - self._tokens,
                        self._max_calls,
                        usage_ratio * 100,
                    )
                return 0.0

            # Not enough tokens — calculate how long until we have enough
            deficit = tokens - self._tokens
            wait_seconds = deficit * (self._period / self._max_calls)

            # After waiting the bucket will have exactly *tokens* available;
            # consume them right now (under lock) so no other thread grabs them.
            self._tokens = 0.0
            self._last_refill = time.monotonic()  # reset the refill clock
            return wait_seconds

    def block_until_ready(self, tokens: int = 1) -> None:
        """Block the current thread until *tokens* are available, then consume them."""
        wait = self.acquire(tokens)
        if wait > 0:
            logger.info(
                "Rate limiter: waiting %.1fs before next call", wait,
            )
            time.sleep(wait)

    def reset(self) -> None:
        """Reset the bucket to full capacity."""
        with self._lock:
            self._tokens = float(self._max_calls)
            self._last_refill = time.monotonic()

    def __repr__(self) -> str:
        return (
            f"RateLimiter(max_calls={self._max_calls}, "
            f"period={self._period}s, "
            f"tokens={self._tokens:.1f})"
        )
