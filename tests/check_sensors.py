"""Проверка чтения температуры CPU и разбора вывода sensors."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.commands.sensors import (  # noqa: E402
    _describe_cpu_temp,
    _is_cpu_temperature,
    _parse_cpu_temp,
    get_cpu_temp,
)

SAMPLE = """ERROR: Can't get value of subfeature power1_cap: Can't read
iwlwifi_1-virtual-0
Adapter: Virtual device
temp1:        +44.0°C

pch_skylake-virtual-0
Adapter: Virtual device
temp1:        +44.5°C

nvme-pci-0500
Adapter: PCI adapter
Composite:    +28.9°C  (low  = -273.1°C, high = +74.8°C)

coretemp-isa-0000
Adapter: ISA adapter
Package id 0:  +60.0°C  (high = +100.0°C, crit = +100.0°C)
Core 0:        +58.0°C  (high = +100.0°C, crit = +100.0°C)
Core 1:        +60.0°C  (high = +100.0°C, crit = +100.0°C)
Core 2:        +56.0°C  (high = +100.0°C, crit = +100.0°C)
Core 3:        +57.0°C  (high = +100.0°C, crit = +100.0°C)
"""

AMD_SAMPLE = """k10temp-pci-00c3
Adapter: PCI adapter
Tctl:         +52.4°C
"""

CASES = [
    ("Intel Package id", _parse_cpu_temp(SAMPLE), 60),
    ("AMD Tctl", _parse_cpu_temp(AMD_SAMPLE), 52),
    ("Pustoy vyvod", _parse_cpu_temp(""), None),
]

print("=== Parsing sensors output ===")
ok = True
for name, got, want in CASES:
    mark = "OK " if got == want else "FAIL"
    if got != want:
        ok = False
    print("  [%s] %s: got=%s want=%s" % (mark, name, got, want))

print("")
print("=== Phrase classification ===")
PHRASES = [
    ("температура процессора", True),
    ("какая температура процессора", True),
    ("сколько градусов у процессора", True),
    ("перегрев компьютера", True),
    ("температура cpu", True),
    ("процессор горячий", True),
    ("какая температура на улице", False),
    ("температура воздуха", False),
    ("какая погода", False),
    ("открой терминал", False),
]
for text, want in PHRASES:
    got = _is_cpu_temperature(text)
    mark = "OK " if got == want else "FAIL"
    if got != want:
        ok = False
    print("  [%s] %-42s -> %s (want %s)" % (mark, repr(text), got, want))

print("")
print("=== Live read ===")
temp = get_cpu_temp()
print("  get_cpu_temp() = %s" % temp)
if temp is not None:
    print("  Otvet: %s" % _describe_cpu_temp(temp))

print("")
print("VSYO ZELYONOYE" if ok else "EST PADENIYA")
