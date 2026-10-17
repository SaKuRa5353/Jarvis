"""LLM через локальный Ollama с persistent историей."""
from src.logger import get_logger

logger = get_logger("llm")
import json

from openai import OpenAI

from config import (
    HISTORY_FILE,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_NUM_THREADS,
    LLM_SYSTEM_PROMPT,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SEC,
    MAX_HISTORY_MESSAGES,
    OLLAMA_BASE_URL,
)


class JarvisLLM:
    def __init__(self, model: str = LLM_MODEL):
        self.client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
            timeout=LLM_TIMEOUT_SEC,
        )
        self.model = model
        self.history = self._load_history()

    def _load_history(self) -> list[dict]:
        if HISTORY_FILE.exists():
            try:
                data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except Exception as e:
                logger.warning("[LLM] не смог прочитать историю: %s", e)
        return []

    def _save_history(self):
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(
            json.dumps(self.history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _trim_history(self):
        if len(self.history) > MAX_HISTORY_MESSAGES:
            self.history = self.history[-MAX_HISTORY_MESSAGES:]

    def clear_history(self):
        self.history = []
        self._save_history()

    def ask(self, text: str, context: str | None = None) -> str | None:
        """Отправляет запрос в LLM.

        context — необязательная строка окружения (например, активное окно).
        Она попадает только в системный промпт текущего запроса и НЕ
        сохраняется в историю, чтобы не накапливать устаревший контекст.
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

        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": answer})
        self._trim_history()
        self._save_history()

        return answer
