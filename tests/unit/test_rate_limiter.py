"""RateLimiter unit tests — token bucket algorithm and thread safety."""

from __future__ import annotations

import threading
import time

import pytest

from quant_agent.data.rate_limiter import RateLimiter


class TestRateLimiterInit:
    def test_default_params(self):
        rl = RateLimiter()
        assert rl.max_calls == 200
        assert rl.period == 60.0

    def test_custom_params(self):
        rl = RateLimiter(max_calls=10, period=5.0)
        assert rl.max_calls == 10
        assert rl.period == 5.0

    def test_invalid_max_calls(self):
        with pytest.raises(ValueError, match="max_calls"):
            RateLimiter(max_calls=0)

    def test_invalid_period(self):
        with pytest.raises(ValueError, match="period"):
            RateLimiter(period=-1)

    def test_invalid_warning_threshold(self):
        with pytest.raises(ValueError, match="warning_threshold"):
            RateLimiter(warning_threshold=0)
        with pytest.raises(ValueError, match="warning_threshold"):
            RateLimiter(warning_threshold=1.5)


class TestRateLimiterAcquire:
    def test_acquire_within_budget(self):
        """Acquiring tokens within budget returns 0 (no wait)."""
        rl = RateLimiter(max_calls=10, period=60.0)
        wait = rl.acquire(1)
        assert wait == 0.0

    def test_acquire_exhausts_tokens(self):
        """After max_calls acquires, next call requires waiting."""
        rl = RateLimiter(max_calls=5, period=60.0)
        for _ in range(5):
            assert rl.acquire(1) == 0.0
        # 6th call should need to wait
        wait = rl.acquire(1)
        assert wait > 0.0

    def test_acquire_zero_tokens(self):
        """Acquiring 0 tokens always returns 0."""
        rl = RateLimiter(max_calls=5, period=60.0)
        assert rl.acquire(0) == 0.0

    def test_acquire_negative_tokens(self):
        """Acquiring negative tokens returns 0."""
        rl = RateLimiter(max_calls=5, period=60.0)
        assert rl.acquire(-1) == 0.0

    def test_wait_time_calculation(self):
        """Wait time should be proportional to deficit."""
        rl = RateLimiter(max_calls=10, period=10.0)
        # Use all tokens
        for _ in range(10):
            rl.acquire(1)
        # 1 token deficit: wait = 1 * (10/10) = 1.0s
        wait = rl.acquire(1)
        assert abs(wait - 1.0) < 0.1

    def test_available_tokens_decreases(self):
        """available_tokens should decrease after acquire."""
        rl = RateLimiter(max_calls=10, period=60.0)
        rl.acquire(3)
        assert rl.available_tokens <= 7.0


class TestRateLimiterReset:
    def test_reset_restores_full_capacity(self):
        rl = RateLimiter(max_calls=5, period=60.0)
        for _ in range(5):
            rl.acquire(1)
        assert rl.available_tokens < 1.0

        rl.reset()
        assert rl.available_tokens == 5.0

    def test_reset_allows_immediate_acquire(self):
        rl = RateLimiter(max_calls=3, period=60.0)
        for _ in range(3):
            rl.acquire(1)
        rl.reset()
        assert rl.acquire(1) == 0.0


class TestRateLimiterRefill:
    def test_tokens_refill_over_time(self):
        """Tokens should gradually refill as time passes."""
        rl = RateLimiter(max_calls=10, period=1.0)  # 10 tokens/sec
        for _ in range(10):
            rl.acquire(1)
        # All tokens used up
        assert rl.available_tokens < 0.01

        # Wait 0.2s → should refill ~2 tokens
        time.sleep(0.2)
        # After refill, we should be able to acquire at least 1
        wait = rl.acquire(1)
        assert wait == 0.0


class TestRateLimiterConcurrency:
    def test_concurrent_acquire_no_overconsumption(self):
        """Multiple threads acquiring tokens should not exceed max_calls
        even without waiting (total consumed <= max_calls)."""
        max_calls = 100
        rl = RateLimiter(max_calls=max_calls, period=60.0)
        consumed = []
        lock = threading.Lock()

        def worker():
            wait = rl.acquire(1)
            if wait == 0.0:
                with lock:
                    consumed.append(1)

        threads = [threading.Thread(target=worker) for _ in range(max_calls + 20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not have consumed more than max_calls
        assert len(consumed) <= max_calls

    def test_block_until_ready(self):
        """block_until_ready should consume a token and return."""
        rl = RateLimiter(max_calls=1, period=1.0)
        rl.block_until_ready(1)
        # All tokens used, next call should block briefly
        start = time.monotonic()
        rl.block_until_ready(1)  # should wait ~1 second
        elapsed = time.monotonic() - start
        assert elapsed >= 0.5  # Allow some tolerance


class TestRateLimiterRepr:
    def test_repr_contains_info(self):
        rl = RateLimiter(max_calls=50, period=30.0)
        s = repr(rl)
        assert "50" in s
        assert "30" in s
