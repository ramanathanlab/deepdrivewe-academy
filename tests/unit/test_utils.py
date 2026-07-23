"""Tests for `deepdrivewe.utils`."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from deepdrivewe.utils import retry_on_exception
from deepdrivewe.utils import wait_for_file


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
