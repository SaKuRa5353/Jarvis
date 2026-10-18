"""LLM через локальный Ollama. История хранится только в текущей сессии."""

from src.logger import get_logger

logger = get_logger("llm")

from openai import OpenAI

from config import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_NUM_THREADS,
    LLM_SYSTEM_PROMPT,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SEC,
    MAX_HISTORY_MESSAGES,
    OLLAMA_BASE_URL,
)

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

    def ask(self, text: str, context: str | None = None) -> str | None:
        """Отправляет запрос в LLM.

        context — необязательная строка окружения (например, активное окно).
        Попадает только в системный промпт текущего запроса.

        Возвращает ответ или None при ошибке.
        """
        system_prompt = LLM_SYSTEM_PROMPT
        if context:
            system_prompt = f"{system_prompt}\n{context}"

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": text})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                extra_body={
                    "options": {"num_thread": LLM_NUM_THREADS},
                },
            )
            raw = response.choices[0].message.content
            if raw is None:
                logger.warning("[LLM] пустой ответ")
                return None
            answer = raw.strip()
        except Exception as e:
            logger.error("[LLM error] %s", e)
            return None

        # Не сохраняем мусорные ответы — модель не будет их копировать.
        if junk_phrase(answer):
            logger.warning("[LLM] мусор в ответе: %s", answer)
            return answer

        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": answer})
        self._trim_history()

        return answer
