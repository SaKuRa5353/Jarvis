"""Проверка импорта голосового контура без runtime-зависимости TTS."""

import subprocess
import sys


def test_voice_modules_import_without_runtime_dependencies() -> None:
    """Импорт голосовых модулей не должен запускать runtime-зависимости."""
    code = """
import importlib.abc
import sys


class BlockRuntimeAudio(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("silero_tts", "vosk", "sounddevice")
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise ModuleNotFoundError("runtime-зависимость заблокирована тестом", name=fullname)
        return None


sys.meta_path.insert(0, BlockRuntimeAudio())
import src.audio
import src.tts
import src.vad
import src.assistant
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
