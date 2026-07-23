"""Tests for `deepdrivewe.utils`."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch

import aiohttp
import pytest

from deepdrivewe.utils import is_transient_error
from deepdrivewe.utils import retry_async
from deepdrivewe.utils import retry_on_exception
from deepdrivewe.utils import wait_for_file


def _response_error(status: int) -> aiohttp.ClientResponseError:
    """Build a ``ClientResponseError`` with the given HTTP status."""
    return aiohttp.ClientResponseError(
        request_info=Mock(),
        history=(),
        status=status,
    )


@pytest.mark.unit
class TestRetryOnException:
    """Covers the `retry_on_exception` decorator."""

    def test_returns_value_when_no_exception(self) -> None:
        @retry_on_exception(wait_time=0)
        def good() -> int:
            return 42

        assert good() == 42

    def test_retries_once_on_exception(self) -> None:
        calls = {'n': 0}

        @retry_on_exception(wait_time=0)
        def flaky() -> str:
            calls['n'] += 1
            if calls['n'] == 1:
                raise RuntimeError('first failure')
            return 'ok'

        with patch('deepdrivewe.utils.time.sleep') as sleep_mock:
            assert flaky() == 'ok'
        assert calls['n'] == 2
        sleep_mock.assert_called_once_with(0)

    def test_reraises_on_second_failure(self) -> None:
        @retry_on_exception(wait_time=0)
        def always_fails() -> None:
            raise ValueError('boom')

        with (
            patch('deepdrivewe.utils.time.sleep'),
            pytest.raises(ValueError, match='boom'),
        ):
            always_fails()


@pytest.mark.unit
class TestWaitForFile:
    """Covers the async NFS-aware `wait_for_file` helper."""

    async def test_returns_immediately_when_present(
        self,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / 'ready.txt'
        target.write_text('hi')
        logger = logging.getLogger('test')
        # Should not raise and should not sleep.
        with patch(
            'deepdrivewe.utils.asyncio.sleep',
        ) as sleep_mock:
            await wait_for_file(target, logger, retries=3, delay=0.01)
        sleep_mock.assert_not_called()

    async def test_retries_then_succeeds(
        self,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / 'late.txt'
        calls = {'n': 0}

        async def fake_sleep(_: float) -> None:
            calls['n'] += 1
            if calls['n'] == 1:
                target.write_text('arrived')

        logger = logging.getLogger('test')
        with patch(
            'deepdrivewe.utils.asyncio.sleep',
            side_effect=fake_sleep,
        ):
            await wait_for_file(target, logger, retries=5, delay=0.01)
        assert target.exists()
        assert calls['n'] >= 1

    async def test_raises_after_exhausting_retries(
        self,
        tmp_path: Path,
    ) -> None:
        missing = tmp_path / 'never.txt'
        logger = logging.getLogger('test')

        async def noop_sleep(_: float) -> None:
            return None

        with (
            patch(
                'deepdrivewe.utils.asyncio.sleep',
                side_effect=noop_sleep,
            ),
            pytest.raises(FileNotFoundError),
        ):
            await wait_for_file(
                missing,
                logger,
                retries=2,
                delay=0.0,
            )


@pytest.mark.unit
class TestIsTransientError:
    """Covers the `is_transient_error` predicate."""

    @pytest.mark.parametrize(
        'exc',
        (
            aiohttp.ClientPayloadError(),
            asyncio.TimeoutError(),
            TimeoutError(),
            _response_error(429),
            _response_error(500),
            _response_error(502),
            _response_error(503),
            _response_error(504),
        ),
    )
    def test_transient(self, exc: BaseException) -> None:
        assert is_transient_error(exc) is True

    @pytest.mark.parametrize(
        'exc',
        (
            _response_error(400),
            _response_error(404),
            _response_error(501),
            ValueError('nope'),
            KeyError('nope'),
        ),
    )
    def test_not_transient(self, exc: BaseException) -> None:
        assert is_transient_error(exc) is False

    def test_cancelled_error_not_transient(self) -> None:
        # CancelledError is a BaseException; it must never be retried.
        assert is_transient_error(asyncio.CancelledError()) is False


@pytest.mark.unit
class TestRetryAsync:
    """Covers the async `retry_async` helper."""

    async def test_returns_first_success_without_sleeping(self) -> None:
        async def ok() -> int:
            return 7

        with patch('deepdrivewe.utils.asyncio.sleep') as sleep_mock:
            assert await retry_async(ok) == 7
        sleep_mock.assert_not_called()

    async def test_retries_transient_then_succeeds(self) -> None:
        calls = {'n': 0}

        async def flaky() -> str:
            calls['n'] += 1
            if calls['n'] < 3:
                raise asyncio.TimeoutError
            return 'done'

        with patch('deepdrivewe.utils.asyncio.sleep') as sleep_mock:
            result = await retry_async(flaky, retries=5, jitter=0.0)
        assert result == 'done'
        assert calls['n'] == 3
        assert sleep_mock.await_count == 2

    async def test_non_transient_raises_immediately(self) -> None:
        calls = {'n': 0}

        async def boom() -> None:
            calls['n'] += 1
            raise ValueError('fatal')

        with (
            patch('deepdrivewe.utils.asyncio.sleep') as sleep_mock,
            pytest.raises(ValueError, match='fatal'),
        ):
            await retry_async(boom, retries=5)
        assert calls['n'] == 1
        sleep_mock.assert_not_called()

    async def test_raises_after_exhausting_retries(self) -> None:
        calls = {'n': 0}

        async def always() -> None:
            calls['n'] += 1
            raise asyncio.TimeoutError

        with (
            patch('deepdrivewe.utils.asyncio.sleep'),
            pytest.raises(asyncio.TimeoutError),
        ):
            await retry_async(always, retries=2, jitter=0.0)
        # 1 initial attempt + 2 retries.
        assert calls['n'] == 3

    async def test_backoff_is_capped_at_max_delay(self) -> None:
        delays: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            delays.append(seconds)

        async def always() -> None:
            raise asyncio.TimeoutError

        with (
            patch(
                'deepdrivewe.utils.asyncio.sleep',
                side_effect=fake_sleep,
            ),
            pytest.raises(asyncio.TimeoutError),
        ):
            await retry_async(
                always,
                retries=5,
                base_delay=1.0,
                max_delay=4.0,
                jitter=0.0,
            )
        # 1, 2, 4, then capped at 4, 4 (no jitter).
        assert delays == [1.0, 2.0, 4.0, 4.0, 4.0]

    async def test_custom_predicate(self) -> None:
        calls = {'n': 0}

        async def flaky() -> str:
            calls['n'] += 1
            if calls['n'] < 2:
                raise ValueError('retry me')
            return 'ok'

        with patch('deepdrivewe.utils.asyncio.sleep'):
            result = await retry_async(
                flaky,
                retries=3,
                predicate=lambda exc: isinstance(exc, ValueError),
                jitter=0.0,
            )
        assert result == 'ok'
        assert calls['n'] == 2
