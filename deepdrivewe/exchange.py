"""Fault-tolerant exchange factory for the Academy framework.

Wraps Academy's :class:`~academy.exchange.cloud.client.HttpExchangeFactory`
so that every transport it creates retries transient network failures
(connection drops, timeouts, and 429/5xx gateway responses) transparently.

Retrying at the transport layer covers *all* exchange traffic — agent
registration, mailbox polling, action dispatch, and result delivery —
rather than a single call site, and a single message ``send`` is far safer
to retry than a whole non-idempotent action. This is the intended home for
connectivity fault tolerance until it lands upstream in Academy.

Example
-------
::

    from deepdrivewe.exchange import RetryingHttpExchangeFactory

    factory = RetryingHttpExchangeFactory(auth_method='globus')
    manager = await Manager.from_exchange_factory(factory=factory, ...)
"""

from __future__ import annotations

import logging
from typing import Any

from academy.exchange.cloud.client import HttpExchangeFactory
from academy.exchange.cloud.client import HttpExchangeTransport
from academy.message import Message

from deepdrivewe.utils import is_transient_error
from deepdrivewe.utils import retry_async

logger = logging.getLogger(__name__)


class RetryingHttpExchangeFactory(HttpExchangeFactory):
    """``HttpExchangeFactory`` whose transports retry transient failures.

    Accepts the same arguments as
    :class:`~academy.exchange.cloud.client.HttpExchangeFactory` plus the
    retry-tuning keywords below.

    Parameters
    ----------
    send_retries : int
        Maximum retries after the initial ``send`` attempt.
    send_base_delay : float
        Initial backoff delay in seconds (doubles each attempt).
    send_max_delay : float
        Upper bound on the backoff delay in seconds.
    """

    def __init__(
        self,
        *args: Any,
        send_retries: int = 5,
        send_base_delay: float = 1.0,
        send_max_delay: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._send_retries = send_retries
        self._send_base_delay = send_base_delay
        self._send_max_delay = send_max_delay

    async def _create_transport(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> HttpExchangeTransport:
        transport = await super()._create_transport(*args, **kwargs)
        return self._wrap_send(transport)

    def _wrap_send(
        self,
        transport: HttpExchangeTransport,
    ) -> HttpExchangeTransport:
        """Bind a retrying ``send`` onto a single transport instance.

        The wrapper is scoped to this transport instance only, unlike a
        class-level monkey-patch which would mutate ``send`` for every
        transport in the process.
        """
        original_send = transport.send
        send_retries = self._send_retries
        send_base_delay = self._send_base_delay
        send_max_delay = self._send_max_delay

        async def send_with_retry(message: Message[Any]) -> None:
            await retry_async(
                lambda: original_send(message),
                retries=send_retries,
                base_delay=send_base_delay,
                max_delay=send_max_delay,
                predicate=is_transient_error,
                logger=logger,
            )

        # Instance-level bind (shadows the class method) — deliberate and
        # scoped, so mypy's method-assignment check does not apply here.
        transport.send = send_with_retry  # type: ignore[method-assign]
        return transport
