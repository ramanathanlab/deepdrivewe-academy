"""Tests for `deepdrivewe.simulation.openmm`.

Focus: `ContactMapCollector.get`, which packs the per-frame contact
maps into a ragged object array. The invariant worth guarding is that
the result stays 1-D (one entry per frame) even when every frame has
the same number of contacts -- otherwise numpy would silently collapse
it to a rectangular 2-D array and downstream ragged handling breaks.
"""

from __future__ import annotations

import numpy as np

from deepdrivewe.simulation.openmm import ContactMapCollector


def _set_frames(
    collector: ContactMapCollector,
    frames: list[np.ndarray],
) -> None:
    """Seed the collector's internal frame buffer directly."""
    collector._contact_maps = list(frames)


def test_get_returns_one_entry_per_frame() -> None:
    """`get` yields a 1-D object array with a slot per frame."""
    collector = ContactMapCollector()
    frames = [
        np.array([0, 1, 2], dtype='int16'),
        np.array([0, 1, 2, 3, 4], dtype='int16'),
    ]
    _set_frames(collector, frames)

    result = collector.get()

    assert result.dtype == object
    assert result.shape == (2,)
    for got, expected in zip(result, frames, strict=True):
        np.testing.assert_array_equal(got, expected)


def test_get_stays_ragged_for_uniform_frame_lengths() -> None:
    """Equal-length frames must not collapse into a 2-D array.

    This is the exact failure mode the explicit construction in
    `get` guards against: `np.array(list, dtype=object)` would build a
    (n_frames, n_contacts) array here instead of a 1-D object array.
    """
    collector = ContactMapCollector()
    frames = [
        np.array([0, 1, 2, 3], dtype='int16'),
        np.array([4, 5, 6, 7], dtype='int16'),
        np.array([8, 9, 10, 11], dtype='int16'),
    ]
    _set_frames(collector, frames)

    result = collector.get()

    assert result.ndim == 1
    assert result.shape == (3,)
    assert result.dtype == object
    for got, expected in zip(result, frames, strict=True):
        np.testing.assert_array_equal(got, expected)


def test_get_empty_collector() -> None:
    """`get` on an untouched collector returns an empty object array."""
    collector = ContactMapCollector()

    result = collector.get()

    assert result.dtype == object
    assert result.shape == (0,)


def test_collect_then_get_roundtrip() -> None:
    """`collect` accumulates frames that `get` returns intact.

    Two atoms placed within the cutoff produce a non-empty contact
    map; `get` should hand back one int16 array per `collect` call.
    """
    collector = ContactMapCollector(cutoff_angstrom=8.0)
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype='float32',
    )

    first = collector.collect(positions)
    second = collector.collect(positions)

    assert first.dtype == np.int16

    result = collector.get()

    assert result.shape == (2,)
    assert result.dtype == object
    np.testing.assert_array_equal(result[0], first)
    np.testing.assert_array_equal(result[1], second)
