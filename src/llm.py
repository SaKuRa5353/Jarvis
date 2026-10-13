"""LLM через локальный Ollama с persistent историей."""

import json

from openai import OpenAI

from config import (
    HISTORY_FILE,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_SYSTEM_PROMPT,
    LLM_TEMPERATURE,
    MAX_HISTORY_MESSAGES,
    OLLAMA_BASE_URL,
)


class JarvisLLM:
    def __init__(self, model: str = LLM_MODEL):
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
        messages = [{"role": "system", "content": LLM_SYSTEM_PROMPT}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": text})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
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
