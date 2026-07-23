"""Tests for `deepdrivewe.exchange`."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import Mock

import pytest

from deepdrivewe.exchange import RetryingHttpExchangeFactory


def _make_factory(**kwargs: Any) -> RetryingHttpExchangeFactory:
    """Build a factory without triggering globus auth resolution.

    A non-default URL keeps ``auth_method`` unset so the constructor
    does not attempt to load Globus credentials.
    """
    return RetryingHttpExchangeFactory(
        url='http://localhost:1234',
        **kwargs,
    )


@pytest.mark.unit
class TestRetryingHttpExchangeFactory:
    """Covers the transport-level retry wrapper."""

    async def test_wrap_send_retries_transient_then_succeeds(self) -> None:
        factory = _make_factory(
            send_retries=3,
            send_base_delay=0.0,
            send_max_delay=0.0,
        )
        calls = {'n': 0}

        async def fake_send(_message: Any) -> None:
            calls['n'] += 1
            if calls['n'] < 2:
                raise asyncio.TimeoutError

        transport = Mock()
        transport.send = fake_send

        wrapped = factory._wrap_send(transport)
        # Same instance is returned with send rebound.
        assert wrapped is transport
        await transport.send(Mock())
        assert calls['n'] == 2

    async def test_wrap_send_propagates_non_transient(self) -> None:
        factory = _make_factory(send_retries=3, send_base_delay=0.0)
        calls = {'n': 0}

        async def fake_send(_message: Any) -> None:
            calls['n'] += 1
            raise ValueError('fatal')

        transport = Mock()
        transport.send = fake_send

        factory._wrap_send(transport)
        with pytest.raises(ValueError, match='fatal'):
            await transport.send(Mock())
        # Non-transient errors are not retried.
        assert calls['n'] == 1

    async def test_create_transport_wraps_send(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        factory = _make_factory(send_retries=1, send_base_delay=0.0)
        calls = {'n': 0}

        async def raw_send(_message: Any) -> None:
            calls['n'] += 1
            if calls['n'] < 2:
                raise asyncio.TimeoutError

        base_transport = Mock()
        base_transport.send = raw_send

        async def fake_super_create(self: Any, *a: Any, **k: Any) -> Mock:
            return base_transport

        # Stand in for the base HttpExchangeFactory transport creation so
        # no real exchange connection is made.
        monkeypatch.setattr(
            'academy.exchange.cloud.client.HttpExchangeFactory'
            '._create_transport',
            fake_super_create,
        )
        transport = await factory._create_transport()
        assert transport is base_transport
        # send was rebound to the retrying wrapper.
        await transport.send(Mock())
        assert calls['n'] == 2
