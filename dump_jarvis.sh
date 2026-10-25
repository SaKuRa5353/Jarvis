#!/usr/bin/env bash
# dump_jarvis.sh — ТОЛЬКО код Джарвиса, без папки КОНКУРЕНТЫ и без мусора.
# Запуск из корня проекта:
#   ./dump_jarvis.sh > /tmp/jarvis.txt ; wc -lc /tmp/jarvis.txt
# Затем прикрепи /tmp/jarvis.txt файлом к сообщению (НЕ вставляй текстом — обрезается).
set -uo pipefail

EXCL=(-path ./.git -o -path ./.venv -o -path ./venv -o -path '*__pycache__*' \
      -o -path ./node_modules -o -path ./models -o -path ./data \
      -o -path './КОНКУРЕНТЫ' -o -path './.idea' -o -path './.vscode' \
      -o -path './.mypy_cache' -o -path './.pytest_cache' -o -path './.ruff_cache' \
      -o -name '*.pyc' -o -name '*.onnx' -o -name '*.wav' -o -name '*.zip' -o -name '*.pt' \
      -o -name 'dump_*.sh' -o -name 'jarvis*.txt')

echo "############ JARVIS DUMP @ $(date -Iseconds) ############"

echo; echo "===== 1. TREE (полное, с размерами) ====="
find . \( "${EXCL[@]}" \) -prune -o -print0 2>/dev/null \
| xargs -0 ls -ld 2>/dev/null | awk '{print $5"\t"$9}' | sort -k2

echo; echo "===== 2. GIT ====="
git rev-parse --abbrev-ref HEAD 2>/dev/null
git log --oneline -20 2>/dev/null
echo "--- status ---"
git status --porcelain 2>/dev/null | head -40

echo; echo "===== 3. ЛОГИ (где реально лежат) ====="
grep -nE 'LOG|\.log|Handler|Path\(' src/logger.py 2>/dev/null
find . -name '*.log' -not -path './.venv/*' -not -path './КОНКУРЕНТЫ/*' 2>/dev/null | head -20
ls -la logs 2>/dev/null

echo; echo "===== 4. requirements.txt — побайтово (ищем артефакты) ====="
tail -c 200 requirements.txt 2>/dev/null | cat -A | tail -8
echo "--- строки, которые pip НЕ сможет разобрать ---"
grep -nE '^\s*[^#\s].*\S\s+#.*"""|"""' requirements.txt 2>/dev/null

echo; echo "===== 5. ОКРУЖЕНИЕ ====="
python -V 2>&1
pip freeze 2>/dev/null | grep -iE 'vosk|silero|torch|onnx|sounddevice|numpy|openai|ollama|dotenv'
echo "--- ollama ---"
systemctl is-active ollama 2>/dev/null; ollama list 2>/dev/null
echo "--- системные тулзы ---"
for b in mpv pactl sensors kdotool spectacle loginctl; do printf '%-12s %s\n' "$b" "$(command -v $b || echo 'НЕ НАЙДЕН')"; done

echo; echo "===== 6. ПОЛНОЕ СОДЕРЖИМОЕ ФАЙЛОВ ====="
find . \( "${EXCL[@]}" \) -prune -o -type f \
  \( -name '*.py' -o -name '*.sh' -o -name '*.toml' -o -name '*.yaml' -o -name '*.yml' \
     -o -name '*.json' -o -name '*.cfg' -o -name '*.ini' -o -name '*.txt' -o -name '*.md' \) \
  -print0 2>/dev/null | sort -z | while IFS= read -r -d '' f; do
    n=$(wc -l < "$f")
    echo; echo "---------- FILE: $f  ($n строк) ----------"
    [ "$n" -gt 1200 ] && head -1200 "$f" && echo "... [ОБРЕЗАНО, всего $n строк]" || cat "$f"
done

echo; echo "===== 7. КАТАЛОГИ, КОТОРЫЕ Я НЕ ЧИТАЛ ====="
du -sh models data 2>/dev/null
ls src/phrases 2>/dev/null | tr '\n' ' '; echo
ls data/tts_cache 2>/dev/null | wc -l

echo; echo "############ END ############"
