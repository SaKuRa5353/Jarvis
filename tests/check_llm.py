#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Диагностика LLM: почему content пустой и сколько длится ответ.

    python tests/check_llm.py
    python tests/check_llm.py qwen2.5:3b qwen3:4b
    python tests/check_llm.py --max-tokens 512
    python tests/check_llm.py --no-warmup

Показывает, какой из способов отключения thinking работает на вашей Ollama.
Запросы идут напрямую через HTTP (urllib), минуя SDK openai, чтобы можно было
передать любые поля и не упереться в валидацию типов.

Проверяются ОБА эндпоинта:
  /api/chat              — родной API Ollama, поле `think` documented
  /v1/chat/completions   — OpenAI-совместимый, им пользуется src/llm.py

Первый запрос к холодной модели включает загрузку в память (2.5-4 ГБ с HDD),
поэтому по умолчанию делается прогрев. Отключить: --no-warmup.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    LLM_MODEL,
    LLM_NUM_THREADS,
    LLM_SYSTEM_PROMPT,
    LLM_TEMPERATURE,
    LLM_TOOLS_MAX_TOKENS,
    OLLAMA_BASE_URL,
)

BASE = OLLAMA_BASE_URL.split("/v1")[0].rstrip("/")
NATIVE = BASE + "/api/chat"
OPENAI = OLLAMA_BASE_URL.rstrip("/") + "/chat/completions"
SHOW = BASE + "/api/show"
PROMPT = "привет"
TIMEOUT = 180

argv = sys.argv[1:]
MAX_TOKENS = LLM_TOOLS_MAX_TOKENS
if "--max-tokens" in argv:
    i = argv.index("--max-tokens")
    MAX_TOKENS = int(argv[i + 1])
    del argv[i:i + 2]
WARMUP = "--no-warmup" not in argv
argv = [a for a in argv if not a.startswith("--")]
MODELS = argv or [LLM_MODEL]


def post(url: str, payload: dict):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, json.loads(r.read().decode("utf-8")), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", "replace")[:300]}, time.perf_counter() - t0
    except Exception as e:
        return -1, {"error": f"{type(e).__name__}: {e}"}, time.perf_counter() - t0


def show_thinking(model: str) -> None:
    code, body, _ = post(SHOW, {"model": model})
    if code != 200:
        print(f"  /api/show: HTTP {code} {str(body.get('error'))[:80]}")
        return
    th = body.get("thinking")
    print(f"  /api/show thinking: {json.dumps(th, ensure_ascii=False) if th else 'нет метаданных'}")
    pi = body.get("model_info") or {}
    for k, v in pi.items():
        if any(x in k for x in ("parameter_size", "quantization", "context_length")):
            print(f"    {k} = {v}")


def row(label: str, code: int, body: dict, dt: float, kind: str) -> None:
    if code != 200:
        print(f"  {label:26} HTTP {code}  {str(body.get('error'))[:90]}")
        return
    if kind == "native":
        msg = body.get("message") or {}
        content = msg.get("content") or ""
        think = msg.get("thinking") or ""
        finish = body.get("done_reason")
        toks = body.get("eval_count")
        dur = (body.get("eval_duration") or 0) / 1e9
        speed = f"{toks / dur:.1f} tok/s" if toks and dur else "-"
    else:
        ch = (body.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        content = msg.get("content") or ""
        think = msg.get("reasoning") or msg.get("reasoning_content") or ""
        finish = ch.get("finish_reason")
        u = body.get("usage") or {}
        toks = u.get("completion_tokens")
        speed = "-"
    mark = "✅" if content.strip() else "❌"
    print(f"  {label:26} {mark} {dt:6.1f}с  content={len(content):4d}  "
          f"thinking={len(think):5d}  finish={finish!s:8} tok={toks!s:5} {speed}")
    if content.strip():
        print(f"      content: {content.strip()[:90]!r}")
    elif think:
        print(f"      thinking: {think.strip()[:90]!r}")


def run_native(model: str, think, label: str) -> None:
    payload = {
        "model": model, "stream": False,
        "messages": [{"role": "system", "content": LLM_SYSTEM_PROMPT},
                     {"role": "user", "content": PROMPT}],
        "options": {"temperature": LLM_TEMPERATURE, "num_predict": MAX_TOKENS,
                    "num_thread": LLM_NUM_THREADS},
    }
    if think is not None:
        payload["think"] = think
    code, body, dt = post(NATIVE, payload)
    row(label, code, body, dt, "native")


def run_openai(model: str, label: str, **extra) -> None:
    payload = {
        "model": model, "stream": False,
        "messages": [{"role": "system", "content": LLM_SYSTEM_PROMPT},
                     {"role": "user", "content": PROMPT}],
        "temperature": LLM_TEMPERATURE, "max_tokens": MAX_TOKENS,
    }
    payload.update(extra)
    code, body, dt = post(OPENAI, payload)
    row(label, code, body, dt, "openai")


def main() -> int:
    print(f"Ollama native : {NATIVE}")
    print(f"OpenAI-compat : {OPENAI}")
    print(f"Промпт {PROMPT!r}   max_tokens/num_predict = {MAX_TOKENS}   timeout = {TIMEOUT}с\n")

    for model in MODELS:
        print(f"═══ {model} ═══")
        show_thinking(model)

        if WARMUP:
            print("  прогрев (загрузка модели в память)...")
            post(NATIVE, {"model": model, "stream": False,
                          "messages": [{"role": "user", "content": "hi"}],
                          "options": {"num_predict": 4}})

        print("  -- родной /api/chat --")
        run_native(model, None, "native base")
        run_native(model, False, "native think=false")
        run_native(model, True, "native think=true")

        print("  -- OpenAI-совместимый /v1 (его использует llm.py) --")
        run_openai(model, "v1 base")
        run_openai(model, "v1 think=false (верхний)", think=False)
        run_openai(model, "v1 extra_body think=false",
                   extra_body={"think": False, "options": {"num_thread": LLM_NUM_THREADS}})
        run_openai(model, "v1 reasoning_effort=none", reasoning_effort="none")
        run_openai(model, "v1 /no_think в тексте",
                   messages=[{"role": "system", "content": LLM_SYSTEM_PROMPT},
                             {"role": "user", "content": PROMPT + " /no_think"}])
        print()

    print("КАК ЧИТАТЬ:")
    print("  ✅ = content непустой, этим способом можно пользоваться")
    print("  Если ✅ только у native think=false, а у /v1 все ❌ —")
    print("  значит OpenAI-слой вашей Ollama не пробрасывает `think`.")
    print("  Тогда вариант: перевести src/llm.py на /api/chat (через пакет ollama,")
    print("  он у вас уже установлен: ollama==0.6.3) или сменить модель на qwen2.5:3b.")
    print()
    print("  Сравните tok/s у разных моделей — при 4 ядрах/8 потоках и HDD")
    print("  разница qwen3:4b против qwen2.5:3b заметна.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
