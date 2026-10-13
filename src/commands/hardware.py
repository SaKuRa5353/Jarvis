"""Команда: характеристики железа.

Парсим системные команды вручную. Вывод — красиво в консоль, голосом — короткое подтверждение.
"""

import re
import subprocess

from src.logger import get_logger

logger = get_logger("commands.hardware")


# ──────────────────────── ЗАПУСК КОМАНД ────────────────────────

def _run(cmd: list[str], timeout: int = 5) -> str:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("Команда %s упала: %s", cmd[0], e)
        return ""


# ──────────────────────── ПАРСИНГ ────────────────────────

def _parse_cpu() -> dict | None:
    """Возвращает dict: {'model': ..., 'cores': ..., 'threads': ...}."""
    lscpu = _run(["lscpu"])
    if not lscpu:
        return None

    data = {}
    for line in lscpu.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            data[key.strip()] = val.strip()

    model = data.get("Model name") or data.get("Имя модели")
    if not model:
        return None
    model = re.sub(r"\s+CPU\s+@.*", "", model)

    threads = data.get("CPU(s)")

    cores = None
    cps = (
        data.get("Core(s) per socket")
        or data.get("Ядер на сокет")
        or data.get("Ядра на сокет")
    )
    socks = (
        data.get("Socket(s)")
        or data.get("Сокетов")
        or data.get("Сокет(ы)")
        or data.get("Сокет")
    )
    try:
        cores = int(cps) * int(socks) if cps and socks else None
    except ValueError:
        pass

    return {"model": model, "cores": cores, "threads": threads}


def _parse_ram() -> dict | None:
    free = _run(["free", "-h"])
    if not free:
        return None
    for line in free.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            if len(parts) >= 3:
                return {"total": parts[1], "used": parts[2]}
    return None


def _parse_gpu() -> list[str] | None:
    lspci = _run(["lspci"])
    if not lspci:
        return None

    gpus = []
    for line in lspci.splitlines():
        if any(k in line for k in ("VGA compatible", "3D controller", "Display controller")):
            if ":" in line:
                desc = line.split(":", 2)[-1].strip()
                desc = re.sub(r"\s*\(rev [^)]+\)", "", desc)
                desc = desc.replace("Corporation ", "").replace("Inc. ", "")
                desc = desc.replace("[AMD/ATI] ", "").replace("Advanced Micro Devices, ", "AMD ")
                gpus.append(desc)

    return gpus if gpus else None


def _parse_disks() -> list[dict] | None:
    """Возвращает список ключевых дисков."""
    df = _run(["df", "-h"])
    if not df:
        return None

    EXCLUDE_PREFIXES = ("tmpfs", "devtmpfs", "efivarfs", "squashfs", "overlay", "none")
    KEY_MOUNTS = ("/", "/mnt/hdd_storage")

    disks = []
    for line in df.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 6:
            continue
        fs, size, avail, mount = parts[0], parts[1], parts[3], parts[5]

        if any(fs.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if "loop" in fs or "squash" in fs:
            continue
        if mount in KEY_MOUNTS:
            disks.append({"mount": mount, "size": size, "avail": avail})

    return disks if disks else None


def _parse_kernel() -> str | None:
    return _run(["uname", "-r"]) or None


# ──────────────────────── КРАСИВЫЙ ВЫВОД ────────────────────────

def _print_pretty(facts: dict):
    """Печатает характеристики в консоль (и в лог-файл через logger)."""
    W = 60
    sep = "═" * W

    lines = [
        "",
        sep,
        " " * 18 + "ХАРАКТЕРИСТИКИ ЖЕЛЕЗА",
        sep,
    ]

    cpu = facts.get("cpu")
    if cpu:
        lines.append("  💻 Процессор")
        lines.append(f"     {cpu['model']}")
        if cpu.get("cores") and cpu.get("threads"):
            lines.append(f"     {cpu['cores']} ядра, {cpu['threads']} потоков")
        lines.append("")

    ram = facts.get("ram")
    if ram:
        lines.append("  🧠 Оперативная память")
        lines.append(f"     {ram['total']} (использовано {ram['used']})")
        lines.append("")

    gpus = facts.get("gpu")
    if gpus:
        lines.append("  🎮 Видеокарты")
        for gpu in gpus:
            lines.append(f"     • {gpu}")
        lines.append("")

    disks = facts.get("disk")
    if disks:
        lines.append("  💾 Диски")
        for d in disks:
            lines.append(f"     {d['mount']:20} {d['size']} всего, {d['avail']} свободно")
        lines.append("")

    kernel = facts.get("kernel")
    if kernel:
        lines.append("  🐧 Ядро Linux")
        lines.append(f"     {kernel}")
        lines.append("")

    lines.append(sep)
    lines.append("")

    # Печатаем через logger — попадёт и в консоль, и в файл
    for line in lines:
        logger.info(line)


# ──────────────────────── HANDLER ────────────────────────

_TRIGGERS = [
    "характеристик", "характеристика", "желез", "железо",
    "что у меня за компьютер", "что за компьютер",
    "конфигураци", "какое у меня железо", "что у меня внутри",
    "расскажи про компьютер", "параметры компьютера",
]


def try_handle(text: str, llm=None):
    if not any(w in text for w in _TRIGGERS):
        return None

    logger.info("Собираю характеристики железа...")

    facts = {
        "cpu": _parse_cpu(),
        "ram": _parse_ram(),
        "gpu": _parse_gpu(),
        "disk": _parse_disks(),
        "kernel": _parse_kernel(),
    }

    _print_pretty(facts)

    # Голосом — только короткое подтверждение
    return "Готово, характеристики в консоли", None