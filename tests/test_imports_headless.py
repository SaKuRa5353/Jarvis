"""Проверка импорта голосового контура без runtime-зависимостей."""

import subprocess
import sys


def test_voice_modules_import_without_runtime_dependencies() -> None:
    """Импорт голосовых модулей не должен запускать runtime-зависимости."""
    code = """
import importlib.abc
import sys


class BlockRuntimeAudio(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("silero_tts", "vosk", "sounddevice", "onnxruntime")
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise ModuleNotFoundError("runtime-зависимость заблокирована тестом", name=fullname)
        return None


sys.meta_path.insert(0, BlockRuntimeAudio())
import src.audio
import src.tts
import src.vad
import src.wakeword
import src.stt
import src.assistant
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_vad_reports_missing_onnxruntime_only_when_used() -> None:
    """VAD импортируется без ONNX, а при создании сессии сообщает о зависимости."""
    code = """
import importlib.abc
import sys


class BlockOnnxRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "onnxruntime" or fullname.startswith("onnxruntime."):
            raise ModuleNotFoundError("onnxruntime заблокирован тестом", name=fullname)
        return None


sys.meta_path.insert(0, BlockOnnxRuntime())
import src.silero_vad

try:
    src.silero_vad.SileroVAD()
except RuntimeError as exc:
    assert "onnxruntime" in str(exc)
else:
    raise AssertionError("ожидалась ясная ошибка об отсутствующем onnxruntime")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_vad_loads_onnxruntime_when_used() -> None:
    """При наличии runtime ONNX-сессия создаётся только при создании VAD."""
    code = """
import sys
import types


class FakeSession:
    def __init__(self, model, providers):
        assert model
        assert providers == ["CPUExecutionProvider"]

    def run(self, *args, **kwargs):
        return None


onnxruntime = types.ModuleType("onnxruntime")
onnxruntime.InferenceSession = FakeSession
sys.modules["onnxruntime"] = onnxruntime

import src.silero_vad

vad = src.silero_vad.SileroVAD()
assert isinstance(vad.session, FakeSession)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
