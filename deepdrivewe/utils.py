"""Workflow utils."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import ParamSpec
from typing import TypeVar

T = TypeVar('T')
P = ParamSpec('P')


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
            except BaseException as e:
                print(
                    f'Exception caught: {e}. \n'
                    f'Retrying after {wait_time} seconds...',
                )
                time.sleep(wait_time)
                return func(*args, **kwargs)

        return wrapper_retry

    return decorator_retry
