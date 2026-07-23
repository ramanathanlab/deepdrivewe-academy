"""Workflow utils."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import os
import random
import time
from collections.abc import Awaitable
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec
from typing import TypeVar

import aiohttp

T = TypeVar('T')
P = ParamSpec('P')

#: HTTP status codes that indicate a transient, retryable failure
#: (gateway/overload errors and rate limiting).
TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


async def wait_for_file(
    path: Path,
    logger: logging.Logger,
    retries: int = 8,
    delay: float = 1.0,
) -> None:
    """Wait for a file to appear on a shared filesystem.

    Forces NFS attribute cache invalidation by listing the
    parent directory, then retries with a delay. Needed on
    HPC clusters where Parsl workers run on different nodes
    and NFS caches may be stale.

    Parameters
    ----------
    path : Path
        The file path to wait for.
    logger : logging.Logger
        Logger for reporting retry attempts.
    retries : int
        Maximum number of retries before raising.
    delay : float
        Initial delay in seconds (doubles each retry).

    Raises
    ------
    FileNotFoundError
        If the file is not found after all retries.
    """
    wait = delay
    for attempt in range(retries):
        with contextlib.suppress(OSError):
            os.listdir(path.parent)
        if path.exists():
            return
        logger.warning(
            f'Waiting for {path} (attempt {attempt + 1}/{retries},'
            f' next retry in {wait:.1f}s)',
        )
        await asyncio.sleep(wait)
        wait *= 2
    raise FileNotFoundError(f'File not found after {retries} retries: {path}')


def is_transient_error(exc: BaseException) -> bool:
    """Return ``True`` if ``exc`` is a transient, retryable network error.

    Covers both failures where the request never received a clean
    response (connection drops, payload errors, timeouts) and gateway
    responses that indicate a temporary server-side condition
    (see :data:`TRANSIENT_STATUS_CODES`).

    Parameters
    ----------
    exc : BaseException
        The exception raised by the operation.

    Returns
    -------
    bool
        Whether the operation should be retried.
    """
    if isinstance(
        exc,
        (
            aiohttp.ClientConnectionError,
            aiohttp.ClientPayloadError,
            # ``asyncio.TimeoutError`` is an alias of the builtin
            # ``TimeoutError`` on Python >= 3.11; list both so the
            # predicate behaves the same on 3.10.
            asyncio.TimeoutError,
            TimeoutError,
        ),
    ):
        return True
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in TRANSIENT_STATUS_CODES
    return False


async def retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.1,
    predicate: Callable[[BaseException], bool] = is_transient_error,
    logger: logging.Logger | None = None,
) -> T:
    """Call an awaitable with exponential backoff, jitter, and a delay cap.

    The single async retry primitive for the project. Retries only when
    ``predicate`` returns ``True`` for the raised exception; anything else
    (including ``asyncio.CancelledError``, which is a ``BaseException`` and
    is never caught here) propagates immediately.

    Parameters
    ----------
    func : Callable[[], Awaitable[T]]
        Zero-argument callable returning the awaitable to run. Called
        afresh on every attempt (e.g. ``lambda: handle.simulate(sim)``).
    retries : int
        Maximum number of retries *after* the initial attempt. Total
        attempts is ``retries + 1``.
    base_delay : float
        Delay before the first retry, in seconds. Doubles each attempt.
    max_delay : float
        Upper bound on the backoff delay, in seconds (applied before
        jitter).
    jitter : float
        Fractional jitter added to each delay, drawn uniformly from
        ``[0, jitter * delay]``. Decorrelates retries across many
        concurrent callers to avoid a thundering herd. Set to ``0`` to
        disable.
    predicate : Callable[[BaseException], bool]
        Returns whether a given exception is retryable. Defaults to
        :func:`is_transient_error`.
    logger : logging.Logger, optional
        If provided, a warning is logged before each retry.

    Returns
    -------
    T
        The result of the first successful call.

    Raises
    ------
    Exception
        The last exception raised once retries are exhausted or the
        exception is not retryable per ``predicate``.
    """
    for attempt in range(retries + 1):
        try:
            return await func()
        except Exception as exc:
            if attempt >= retries or not predicate(exc):
                raise
            delay = min(base_delay * 2.0**attempt, max_delay)
            if jitter:
                delay += random.uniform(0, jitter * delay)
            if logger is not None:
                logger.warning(
                    'Transient error %r, retrying in %.1fs (attempt %d/%d)',
                    exc,
                    delay,
                    attempt + 1,
                    retries,
                )
            await asyncio.sleep(delay)
    raise AssertionError('unreachable')  # pragma: no cover


def retry_on_exception(
    wait_time: int,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Retry a function if an exception is raised.

    Parameters
    ----------
    wait_time: int
        Time to wait before retrying the function.
    """

    def decorator_retry(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper_retry(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            # Narrow to ``Exception`` so control-flow signals such as
            # ``KeyboardInterrupt`` and ``asyncio.CancelledError`` are not
            # swallowed and retried.
            except Exception as e:
                print(
                    f'Exception caught: {e}. \n'
                    f'Retrying after {wait_time} seconds...',
                )
                time.sleep(wait_time)
                return func(*args, **kwargs)

        return wrapper_retry

    return decorator_retry
