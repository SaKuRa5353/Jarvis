"""Проверка импорта голосового контура без runtime-зависимости TTS."""

import subprocess
import sys


def test_tts_and_assistant_import_without_silero_tts() -> None:
    """Импорт модулей не должен загружать Silero или запускать синтез."""
    code = """
import importlib.abc
import sys


class BlockSilero(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "silero_tts" or fullname.startswith("silero_tts."):
            raise ModuleNotFoundError("silero_tts заблокирован тестом", name=fullname)
        return None


sys.meta_path.insert(0, BlockSilero())
import src.tts
import src.assistant
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
