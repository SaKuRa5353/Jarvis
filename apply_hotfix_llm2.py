#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HOTFIX v2: LLM возвращает пустой content и думает по 30-47 секунд.

ДИАГНОЗ (подтверждён `ollama show qwen3-jarvis` и вашим логом):
    architecture  qwen3, 4.0B, Q4_K_M
    capabilities  completion, tools, thinking -> levels true, **default true**
    parameters    num_predict 150
    system        «... НЕ размышляй, не пиши Thinking ...»   <- не помогает

  Thinking у Qwen3 отключается полем `think` в запросе, а не текстом промпта:
   - `PARAMETER think false` в Modelfile -> "Error: unknown parameter 'think'"
     (ollama/ollama#14809)
   - `SYSTEM /no_think` в Modelfile затирается системным промптом клиента,
     а llm.py всегда шлёт LLM_SYSTEM_PROMPT (тот же тикет)
  Поэтому thinking съедает весь бюджет max_tokens=128, в `content` приходит '',
  рассуждение уходит в `reasoning`/`thinking`, finish_reason='length', 18-47 с.

ЧТО ДЕЛАЕТ ПАТЧ
  config.py   LLM_THINKING=False, LLM_THINK_MODE, LLM_EMPTY_FALLBACK
  src/llm.py  think=False кладётся В EXTRA_BODY ВЕРХНИМ УРОВНЕМ: SDK openai
              подмешивает extra_body в JSON как есть, и Ollama видит поле `think`
              там, где ожидает. Вариант "effort" шлёт reasoning_effort="none",
              "none" не меняет запрос. options.num_thread не теряется.
              Читает msg.reasoning; при пустом content подменяет ответ готовой
              фразой и не пишет его в историю.
  src/tts.py  speak('') не синтезирует тишину

ТРЕБУЕТ применённого apply_qw.py (quick wins). Поверх apply_hotfix_llm.py (v1)
НЕ ставится — скрипт это определит и попросит сначала откатить v1.

    python apply_hotfix_llm2.py --check
    python apply_hotfix_llm2.py
    python apply_hotfix_llm2.py --revert
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DRY = "--check" in sys.argv
REVERT = "--revert" in sys.argv

R: list[tuple[str, str, str]] = []


def rep(path: str, old: str, new: str) -> None:
    R.append((path, old, new))


# ═══════════════════════════════ config.py ═══════════════════════════════
CONFIG_BLOCK = (
    "# Thinking/reasoning у Qwen3. Модель qwen3-jarvis собрана с thinking.default=true,\n"
    "# поэтому Ollama отдаёт текст в поле reasoning/thinking, а content приходит ПУСТЫМ.\n"
    "# Текстом промпта это не отключается — только полем `think` в запросе.\n"
    "#   think  -> extra_body={'think': False}   (основной способ)\n"
    "#   effort -> reasoning_effort='none'\n"
    "#   none   -> запрос не меняется\n"
    "# Что работает на вашей Ollama — покажет  python tests/check_llm.py\n"
    "LLM_THINKING = False\n"
    'LLM_THINK_MODE = "think"\n'
    "\n"
    "# Чем заменять пустой/оборванный ответ LLM (ключ готовой фразы).\n"
    'LLM_EMPTY_FALLBACK = "not_listened"\n'
    "\n"
)

rep("config.py", "LLM_SYSTEM_PROMPT = (\n", CONFIG_BLOCK + "LLM_SYSTEM_PROMPT = (\n")

# ═══════════════════════════════ src/llm.py ══════════════════════════════
rep("src/llm.py",
    "from config import (\n    LLM_MAX_TOKENS,\n    LLM_MODEL,\n",
    "from config import (\n"
    "    LLM_EMPTY_FALLBACK,\n"
    "    LLM_MAX_TOKENS,\n"
    "    LLM_MODEL,\n"
    "    LLM_THINKING,\n"
    "    LLM_THINK_MODE,\n")

REQUEST_OLD = (
    "        request = dict(\n"
    "            model=self.model,\n"
    "            messages=messages,\n"
    "            temperature=LLM_TEMPERATURE,\n"
    "            max_tokens=max_tokens,\n"
    "            tools=TOOLS,\n"
    '            tool_choice="auto",\n'
    '            extra_body={"options": {"num_thread": LLM_NUM_THREADS}},\n'
    "        )\n"
    "        finish_reason: str | None = None\n"
)

THINK_BODY = (
    "        # Think-параметры собираем ОТДЕЛЬНО от dict(): так не теряется\n"
    "        # extra_body с options.num_thread.\n"
    '        extra_body: dict = {"options": {"num_thread": LLM_NUM_THREADS}}\n'
    "        extra: dict = {}\n"
    "        if not LLM_THINKING:\n"
    '            if LLM_THINK_MODE == "think":\n'
    "                # SDK openai подмешивает extra_body в тело JSON как есть ->\n"
    '                # Ollama видит поле "think" верхнего уровня, где и ожидает.\n'
    '                extra_body["think"] = False\n'
    '            elif LLM_THINK_MODE == "effort":\n'
    '                extra["reasoning_effort"] = "none"\n'
    "\n"
    "        request = dict(\n"
    "            model=self.model,\n"
    "            messages=messages,\n"
    "            temperature=LLM_TEMPERATURE,\n"
    "            max_tokens=max_tokens,\n"
    "            tools=TOOLS,\n"
    '            tool_choice="auto",\n'
    "            extra_body=extra_body,\n"
    "            **extra,\n"
    "        )\n"
    "        finish_reason: str | None = None\n"
    '        reasoning_text = ""\n'
)

rep("src/llm.py", REQUEST_OLD, THINK_BODY)

rep("src/llm.py",
    '                finish_reason = getattr(choice, "finish_reason", None)\n',
    '                finish_reason = getattr(choice, "finish_reason", None)\n'
    '                reasoning_text = getattr(msg, "reasoning", None) or getattr(\n'
    '                    msg, "reasoning_content", None\n'
    '                ) or ""\n')

EMPTY_OLD = (
    "        answer = raw.strip()\n"
    "\n"
    "        # Ответ, обрезанный по max_tokens, озвучивать как полноценный нельзя:\n"
    "        # половина фразы звучит как ошибка распознавания.\n"
    '        if finish_reason == "length":\n'
    "            logger.warning(\n"
    '                "[LLM] ответ оборван по max_tokens=%d: %r",\n'
    '                request.get("max_tokens", 0), answer[-60:],\n'
    "            )\n"
)

EMPTY_NEW = (
    "        answer = raw.strip()\n"
    "\n"
    "        # Пустой ответ = thinking съел весь бюджет, content не отдан.\n"
    "        # Озвучивать нечего: без подмены TTS синтезирует тишину (~1.5 с).\n"
    "        if not answer:\n"
    "            logger.warning(\n"
    '                "[LLM] пустой ответ (reasoning=%d симв., finish=%s) — "\n'
    '                "замена на фразу %s",\n'
    "                len(reasoning_text), finish_reason, LLM_EMPTY_FALLBACK,\n"
    "            )\n"
    "            if reasoning_text:\n"
    '                logger.debug("[LLM reasoning] %s", reasoning_text[:300])\n'
    '            return "Не расслышал", LLM_EMPTY_FALLBACK\n'
    "\n"
    "        # Ответ, обрезанный по max_tokens, озвучивать как полноценный нельзя:\n"
    "        # половина фразы звучит как ошибка распознавания.\n"
    '        if finish_reason == "length":\n'
    "            logger.warning(\n"
    '                "[LLM] ответ оборван по max_tokens=%s: %r",\n'
    '                request.get("max_tokens"), answer[-60:],\n'
    "            )\n"
)

rep("src/llm.py", EMPTY_OLD, EMPTY_NEW)

# ═══════════════════════════════ src/tts.py ══════════════════════════════
rep("src/tts.py",
    "def speak(text: str, phrase_key: str | None = None):\n",
    "def speak(text: str, phrase_key: str | None = None):\n"
    "    if not text or not text.strip():\n"
    '        logger.warning("[TTS] пустой текст — не озвучиваю")\n'
    "        return\n")

# ------------------------------------------------------------------ движок --
TARGETS = sorted({f for f, _, _ in R})

NEED = {
    "config.py": ["LLM_THINKING = False", 'LLM_THINK_MODE = "think"', "LLM_EMPTY_FALLBACK"],
    "src/llm.py": ['extra_body["think"] = False', 'extra["reasoning_effort"] = "none"',
                   "reasoning_text", "LLM_EMPTY_FALLBACK", "**extra,"],
    "src/tts.py": ["пустой текст — не озвучиваю"],
}

# признаки применённого хотфикса v1 — поверх него v2 не ставится
V1_MARKERS = {
    "config.py": "# Thinking/reasoning у Qwen3. При включённом reasoning Ollama отдаёт текст",
    "src/llm.py": 'request["extra_body"] = {',
}


def detect_v1() -> list[str]:
    found = []
    for rel, marker in V1_MARKERS.items():
        p = ROOT / rel
        if p.exists() and marker in p.read_text(encoding="utf-8"):
            found.append(rel)
    return found


def apply() -> int:
    v1 = detect_v1()
    if v1:
        print("\u274C Обнаружен хотфикс v1 в: " + ", ".join(v1))
        print("   v2 поверх v1 не ставится: якоря накладываются друг на друга.")
        print("   Сначала откатите v1, потом примените v2:")
        print("       python apply_hotfix_llm.py --revert")
        print("       python apply_hotfix_llm2.py")
        return 1

    errors = 0
    for rel in TARGETS:
        p = ROOT / rel
        if not p.exists():
            print(f"\u26A0 {rel}: файл не найден")
            errors += 1
            continue

        orig = p.read_text(encoding="utf-8")
        text = orig
        done = already = 0

        for f, old, new in R:
            if f != rel:
                continue
            n = text.count(old)
            if n == 1:
                text = text.replace(old, new, 1)
                done += 1
            elif n > 1:
                head = old.strip().splitlines()[0][:58]
                print(f"\u274C {rel}: якорь найден {n} раза -> {head!r}")
                errors += 1
            elif new and new in text:
                already += 1
            else:
                head = old.strip().splitlines()[0][:58]
                print(f"\u274C {rel}: якорь НЕ НАЙДЕН -> {head!r}")
                errors += 1

        missing = [x for x in NEED.get(rel, []) if x not in text]
        if missing:
            print(f"\u274C {rel}: после правок не хватает {missing}")
            errors += 1
            continue

        lines = text.split("\n")
        dups = {lines[i].strip() for i in range(1, len(lines))
                if lines[i].strip() and lines[i] == lines[i - 1]}
        if dups:
            print(f"\u274C {rel}: дублирующиеся строки подряд {sorted(dups)[:3]}")
            errors += 1
            continue

        if errors:
            continue

        if DRY:
            state = "OK" if text != orig else "уже применено"
            print(f"  \u2713 {rel}: {state} (новых правок {done}, уже было {already})")
        elif text != orig:
            bak = p.with_name(p.name + ".hotbak")
            if not bak.exists():          # не затираем исходник при повторном запуске
                shutil.copy2(p, bak)
            p.write_text(text, encoding="utf-8")
            print(f"  \u2713 {rel}: применено {done}, уже было {already}")
        else:
            print(f"  \u00B7 {rel}: без изменений (уже применено {already})")

    print()
    if DRY:
        print("DRY RUN — ничего не записано" if not errors else "DRY RUN с ошибками — не применять")
    else:
        print("Записано. Бэкапы: *.hotbak" if not errors else "ЕСТЬ ОШИБКИ — проверьте вывод выше")
    return 1 if errors else 0


def revert() -> int:
    n = 0
    for bak in sorted(ROOT.rglob("*.hotbak")):
        target = bak.with_name(bak.name[: -len(".hotbak")])
        shutil.move(str(bak), str(target))
        print(f"  \u21A9 {target.relative_to(ROOT)}")
        n += 1
    print(f"Откачено: {n}")
    return 0


def verify() -> int:
    bad = 0
    files = [ROOT / r for r in TARGETS if (ROOT / r).suffix == ".py"]
    for f in files:
        r = subprocess.run([sys.executable, "-m", "py_compile", str(f)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            bad += 1
            print(f"\u274C {f}: {r.stderr.strip()[:400]}")
    print(f"\npy_compile: {len(files) - bad}/{len(files)} OK")
    return 1 if bad else 0


if __name__ == "__main__":
    if REVERT:
        raise SystemExit(revert())
    code = apply()
    if not DRY and code == 0:
        code |= verify()
    raise SystemExit(code)
