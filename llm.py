"""
LLM-модуль Джарвиса через локальный Ollama.

- Persistent история диалога (data/history.json)
- Системный промпт для персонажа Джарвиса
- Обрезка истории (последние N сообщений)
"""

import json
from pathlib import Path
from openai import OpenAI


# ──────────────────────── КОНСТАНТЫ ────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "qwen2.5:3b"

HISTORY_FILE = Path("data/history.json")
MAX_HISTORY_MESSAGES = 20

SYSTEM_PROMPT = """Ты — Джарвис, голосовой ассистент пользователя.
Отвечай ОЧЕНЬ кратко — одно предложение, максимум два.
По-русски, дружелюбно и по делу.
Не используй списки, markdown и форматирование — твой ответ будет озвучен голосом.
Если не знаешь точного ответа — честно скажи об этом, не выдумывай.
Обращайся к пользователю на «ты»."""


# ──────────────────────── КЛАСС ────────────────────────

class JarvisLLM:
    def __init__(self, model: str = MODEL):
        self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        self.model = model
        self.history = self._load_history()

    def _load_history(self) -> list[dict]:
        if HISTORY_FILE.exists():
            try:
                data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except Exception as e:
                print(f"[LLM] не смог прочитать историю: {e}")
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

    def ask(self, text: str) -> str | None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": text})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=100,
            )
            raw = response.choices[0].message.content
            if raw is None:
                print("[LLM] пустой ответ")
                return None
            answer = raw.strip()
        except Exception as e:
            print(f"[LLM error] {e}")
            return None

        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": answer})
        self._trim_history()
        self._save_history()

        return answer