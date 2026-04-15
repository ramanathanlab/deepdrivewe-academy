"""Smoke tests for the deepdrivewe package."""

from __future__ import annotations

import deepdrivewe


def test_import() -> None:
    """Package imports cleanly and exposes a version string."""
    assert deepdrivewe.__version__
