"""Регрессии контракта фиксированного размера аудиокадров."""

import numpy as np

from src.audio import BroadcastBuffer


def test_subscriber_splits_large_block_into_exact_frames():
    broadcast = BroadcastBuffer()
    subscriber = broadcast.subscribe()
    block = np.arange(1280, dtype=np.int16).tobytes()
    broadcast.push(block)

    first = subscriber.read(512, timeout=0.01)
    second = subscriber.read(512, timeout=0.01)
    third = subscriber.read(256, timeout=0.01)

    assert [len(first), len(second), len(third)] == [512, 512, 256]
    np.testing.assert_array_equal(
        np.concatenate([first, second, third]), np.arange(1280, dtype=np.int16)
    )
