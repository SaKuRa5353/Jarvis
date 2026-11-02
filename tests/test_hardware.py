"""Регрессия зависания сбора характеристик: без PCI-опроса и оборудования."""

from __future__ import annotations

from pathlib import Path

from src.commands import hardware


def test_gpu_reads_sysfs_without_commands(tmp_path: Path, monkeypatch) -> None:
    """G состоит из DRM-карт, разъёмы card0-HDMI-A-1 не являются картами."""
    device = tmp_path / "card0" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n", encoding="utf-8")
    (device / "device").write_text("0x67ef\n", encoding="utf-8")
    driver = tmp_path / "amdgpu"
    driver.mkdir()
    (device / "driver").symlink_to(driver, target_is_directory=True)
    (tmp_path / "card0-HDMI-A-1").mkdir()

    def forbidden(*args, **kwargs):
        """Любой subprocess в GPU-ветке нарушает защиту от зависания."""
        raise AssertionError("Нельзя запускать lspci")

    monkeypatch.setattr(hardware, "_run", forbidden)
    assert hardware._parse_gpu(tmp_path) == ["AMD 0x67ef (драйвер amdgpu)"]


def test_missing_gpu_attributes(tmp_path: Path) -> None:
    """При отсутствии устройств G пусто, обработчик возвращает None."""
    assert hardware._parse_gpu(tmp_path) is None
    (tmp_path / "card2").mkdir()
    assert hardware._parse_gpu(tmp_path) is None


def test_hardware_command_completes_without_lspci(monkeypatch, caplog, tmp_path) -> None:
    """План включает прежнее подтверждение и печатает доступные характеристики."""
    def fake_run(cmd: list[str], timeout: int = 5) -> str:
        """Вывод системных команд; PCI-опрос не разрешён."""
        assert cmd[0] != "lspci"
        return "проверка" if cmd[0] == "uname" else ""

    monkeypatch.setattr(hardware, "_run", fake_run)
    parse_gpu = hardware._parse_gpu
    monkeypatch.setattr(hardware, "_parse_gpu", lambda: parse_gpu(tmp_path))
    with caplog.at_level("INFO"):
        result = hardware.try_handle("характеристики железа")
    assert result == ("Готово, характеристики в консоли", None, "done")