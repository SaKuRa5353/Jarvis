"""Мастер настройки микрофона: python main.py --setup

Показывает только пригодные устройства ввода (реальные карты, без ALSA-виртуалов
и HDMI без входа), пишет устойчивое имя устройства в .env и предлагает тестовую
запись 3 секунды с пиковой амплитудой.

Зачем: DEVICE был захардкожен как 4 в config.py и отдельным литералом
в tests/check_stt.py — на другой машине проект не заводился.
"""

from __future__ import annotations

from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
ENV_KEYS = ("JARVIS_DEVICE_NAME", "JARVIS_HOSTAPI", "JARVIS_DEVICE")

# ALSA-виртуалы и агрегаторы: их в списке sounddevice больше десятка, толку нет.
_SKIP_NAMES = {
    "sysdefault", "front", "surround40", "surround51", "surround71",
    "hdmi", "lavrate", "samplerate", "speexrate", "speex", "upmix",
    "vdownmix", "dmix", "pipewire", "pulse", "default",
}


def list_input_devices() -> list[tuple[int, str, int, str]]:
    """(индекс, имя, число входов, host API) — только устройства ввода."""
    import sounddevice as sd

    out: list[tuple[int, str, int, str]] = []
    for i, d in enumerate(sd.query_devices()):
        if int(d.get("max_input_channels", 0)) <= 0:
            continue
        name = str(d.get("name", ""))
        base = name.split(":")[0].strip().lower()
        if base in _SKIP_NAMES or name.strip().lower() in _SKIP_NAMES:
            continue
        hostapi = d.get("hostapi")
        hostapi_name = str(hostapi)
        if isinstance(hostapi, int):
            hostapis = sd.query_hostapis()
            if 0 <= hostapi < len(hostapis):
                hostapi_name = str(hostapis[hostapi].get("name", hostapi))
        out.append((i, name, int(d["max_input_channels"]), hostapi_name))
    return out


def _print_devices(devs: list[tuple[int, str, int]]) -> None:
    print("\nПригодные устройства ввода:\n")
    for idx, name, ch, hostapi in devs:
        print(f"  [{idx:>2}] {name}  ({ch} in, {hostapi})")
    print()


def _write_env(device: int, name: str, hostapi: str) -> None:
    lines: list[str] = []
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    lines = [
        ln for ln in lines
        if not any(ln.strip().startswith(f"{key}=") for key in ENV_KEYS)
    ]
    lines.extend([
        f"JARVIS_DEVICE_NAME={name}",
        f"JARVIS_HOSTAPI={hostapi}",
        f"JARVIS_DEVICE={device}",
    ])
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✅ Имя устройства и host API записаны в {ENV_FILE}")
    print("   (.env уже в .gitignore — в репозиторий не попадёт)")


def _record_test(device: int) -> bool:
    print("\n🎤 Скажите что-нибудь 3 секунды...")
    try:
        import numpy as np
        import sounddevice as sd

        from config import SAMPLE_RATE

        rec = sd.rec(int(3 * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                     channels=1, dtype="int16", device=device)
        sd.wait()
        amp = int(np.abs(rec).max())
    except Exception as e:
        print(f"   ⚠️ Запись не удалась: {e}")
        return False

    print(f"   Max amplitude: {amp}")
    if amp < 500:
        print("   ❌ Слишком тихо. pavucontrol → Input Devices → поднять усиление.")
        return False
    print("   ✅ Микрофон слышит вас.")
    return True


def run_setup(list_only: bool = False) -> int:
    try:
        import sounddevice as sd  # noqa: F401
    except Exception as e:
        print(f"❌ sounddevice недоступен: {e}")
        print("   pip install sounddevice   (нужен системный PortAudio)")
        return 1

    devs = list_input_devices()
    if not devs:
        print("❌ Устройств ввода не найдено. Проверьте микрофон и PortAudio.")
        return 1

    _print_devices(devs)
    if list_only:
        return 0

    try:
        from config import DEVICE as current
        print(f"Сейчас в конфиге: DEVICE={current}\n")
    except Exception:
        current = None

    valid = {str(i) for i, _, _, _ in devs}
    raw = input("Номер микрофона (Enter = оставить текущий): ").strip()
    if not raw:
        if current is None or current < 0:
            print("Текущего значения нет — укажите номер.")
            return 1
        print(f"Оставляем DEVICE={current}")
        chosen = int(current)
    elif raw not in valid:
        print(f"❌ {raw!r} не в списке {sorted(valid, key=int)}")
        return 1
    else:
        chosen = int(raw)
        name, hostapi = next((name, hostapi) for i, name, _, hostapi in devs if i == chosen)
        _write_env(chosen, name, hostapi)

    if input("\nПроверить запись 3 секунды? [Y/n] ").strip().lower() in ("", "y", "д"):
        _record_test(chosen)

    print("\nГотово. Запуск: python main.py")
    return 0
