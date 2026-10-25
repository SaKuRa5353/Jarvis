"""LLM через локальный Ollama. История хранится только в текущей сессии.

Поддерживает function calling (tools): модель выбирает функцию и
аргументы, выполняет их Python по белым спискам (src/llm_tools.py).

Для раннего старта TTS есть режим стриминга: ask(..., on_delta=cb)
отдаёт куски текстового ответа в колбэк по мере генерации; tool-вызовы
при этом накапливаются и обрабатываются как обычно (они приходят
одним куском в конце — подтверждения всё равно готовые фразы).
"""

from types import SimpleNamespace

from openai import OpenAI

from config import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_NUM_THREADS,
    LLM_SYSTEM_PROMPT,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SEC,
    LLM_TOOLS_HINT,
    LLM_TOOLS_MAX_TOKENS,
    MAX_HISTORY_MESSAGES,
    OLLAMA_BASE_URL,
)

from src.llm_tools import TOOLS, execute_tool
from src.logger import get_logger
from src.metrics import timed

logger = get_logger("llm")

# Ответы-пустышки: не добавляем их в историю, чтобы модель не копировала
# собственную ошибку из контекста. Значение — ключ готовой фразы, которой
# ассистент заменит синтез.
JUNK_ANSWERS = {
    "не расслышал": "not_listened",
    "не придумывай факты": "not_listened",
}


def junk_phrase(answer: str) -> str | None:
    """Ключ готовой фразы, если ответ — мусор, иначе None."""
    key = answer.lower().strip().rstrip(".,!?;:")
    return JUNK_ANSWERS.get(key)


class JarvisLLM:
    def __init__(self, model: str = LLM_MODEL):
        self.client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
            timeout=LLM_TIMEOUT_SEC,
        )
        self.model = model
        self.history: list[dict] = []

    def clear_history(self):
        self.history = []

    def _trim_history(self):
        if len(self.history) > MAX_HISTORY_MESSAGES:
            self.history = self.history[-MAX_HISTORY_MESSAGES:]

    @timed("llm.ask")
    def ask(
        self,
        text: str,
        context: str | None = None,
        on_delta=None,
    ) -> tuple[str, str | None] | None:
        """Отправляет запрос в LLM с tools (function calling).

        context — необязательная строка окружения (например, активное окно).
        Попадает только в системный промпт текущего запроса.

        on_delta(chunk) — если задан, текстовый ответ запрашивается
        в режиме стриминга, и каждый кусок текста уходит в колбэк по мере
        генерации (ранний старт TTS).

        Возвращает (ответ, ключ_готовой_фразы) или None при ошибке.
        Если модель вызвала tool — действие уже выполнено, ответ —
        текст подтверждения.
        """
        system_prompt = f"{LLM_SYSTEM_PROMPT}\n{LLM_TOOLS_HINT}"
        if context:
            system_prompt = f"{system_prompt}\n{context}"

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": text})

        # tools передаются всегда → фактический лимит LLM_TOOLS_MAX_TOKENS.
        # LLM_MAX_TOKENS применяется, только если tools отключат.
        # (До quick wins LLM_MAX_TOKENS не использовался вовсе.)
        max_tokens = LLM_TOOLS_MAX_TOKENS if TOOLS else LLM_MAX_TOKENS

        request = dict(
            model=self.model,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=max_tokens,
            tools=TOOLS,
            tool_choice="auto",
            extra_body={"options": {"num_thread": LLM_NUM_THREADS}},
        )
        finish_reason: str | None = None

        try:
            if on_delta is None:
                choice = self.client.chat.completions.create(**request).choices[0]
                msg = choice.message
                tool_calls = getattr(msg, "tool_calls", None)
                raw = msg.content
                finish_reason = getattr(choice, "finish_reason", None)
            else:
                raw, tool_calls, finish_reason = self._ask_streaming(
                    on_delta=on_delta, **request
                )

            if tool_calls:
                return self._run_tool(tool_calls[0])
        except Exception as e:
            logger.error("[LLM error] %s", e)
            return None

        if raw is None:
            logger.warning("[LLM] пустой ответ")
            return None
        answer = raw.strip()

        # Ответ, обрезанный по max_tokens, озвучивать как полноценный нельзя:
        # половина фразы звучит как ошибка распознавания.
        if finish_reason == "length":
            logger.warning(
                "[LLM] ответ оборван по max_tokens=%d: %r",
                request.get("max_tokens", 0), answer[-60:],
            )

        # Не сохраняем мусорные ответы — модель не будет их копировать.
        junk = junk_phrase(answer)
        if junk:
            logger.warning("[LLM] мусор в ответе: %s", answer)
            return answer, junk

        self._append_history(text, answer)
        return answer, None

    def _ask_streaming(self, on_delta, **request) -> tuple[str | None, list | None, str | None]:
        """Стриминговый запрос: текст уходит в on_delta, tool-вызовы копятся.

        Возвращает (текст_ответа, [tool_call] или None). Куски tool-вызова
        (id, имя, аргументы) приходят порознь и склеиваются по index.
        """
        parts: list[str] = []
        calls: dict[int, dict] = {}
        finish: str | None = None

        for chunk in self.client.chat.completions.create(**request, stream=True):
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if getattr(choice, "finish_reason", None):
                finish = choice.finish_reason
            if delta is None:
                continue
            if delta.content:
                parts.append(delta.content)
                on_delta(delta.content)
            for tc in getattr(delta, "tool_calls", None) or ():
                slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function is not None:
                    if tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function.arguments:
                        slot["args"] += tc.function.arguments

        if calls:
            first = calls[min(calls)]
            call = SimpleNamespace(
                id=first["id"],
                function=SimpleNamespace(name=first["name"], arguments=first["args"]),
            )
            return None, [call], finish
        return "".join(parts), None, finish

    def _run_tool(self, call) -> tuple[str, str | None]:
        """Выполняет выбранный моделью tool."""
        result = execute_tool(call.function.name, call.function.arguments)
        if result is None:
            logger.warning(
                "[tool] %s(%s) — невалидные аргументы",
                call.function.name, call.function.arguments,
            )
            return "Не расслышал", "not_listened"
        answer, phrase_key = result
        logger.info("[tool] %s(%s) → %s", call.function.name, call.function.arguments, answer)
        # Подтверждения действий НЕ пишем в историю: 3B-модель
        # начинает копировать их («открой X» → «Открываю вконтакт»)
        # на следующем запросе.
        return answer, phrase_key

    def _append_history(self, text: str, answer: str):
        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": answer})
        self._trim_history()
