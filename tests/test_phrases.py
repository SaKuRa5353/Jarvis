"""Unit-тесты готовых фраз и слогана resolve_phrase_key."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.phrases import get_phrase_path, resolve_phrase_key  # noqa: E402


def test_known_phrases_have_wav():
    for key in (
        "welcome", "opening", "closing", "done", "llm_thinking",
        "not_listened", "not_understood", "error", "lock_screen",
    ):
        path = get_phrase_path(key)
        assert path is not None, f"нет фразы {key}"
        assert path.exists(), f"нет файла для фразы {key}: {path}"


def test_resolve_phrase_key_off():
    import config
    from unittest.mock import patch

    with patch.object(config, "CONFIRM_SOUND", "off"):
        assert resolve_phrase_key("welcome") is None
