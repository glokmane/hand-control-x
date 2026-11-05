# Copyright (c) 2026 Hand Control X authors. All rights reserved.
"""
AI assistant for Hand Control X.
Cloud: OpenAI-compatible Chat Completions API.
Local: Ollama OpenAI-compatible endpoint.
No additional Python packages are required.
"""
import json
import os
import re
import threading
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from urllib.parse import quote_plus, urlparse

HERE = Path(__file__).resolve().parent
SETTINGS_FILE = HERE / "ai_settings.json"
MAX_STEPS = 6
TIMEOUT_DEFAULT = 90


def _load_settings():
    defaults = {
        "enabled": True,
        "provider": "auto",
        "deepseek": {
            "model": "deepseek-chat",
            "api_key_env": "DEEPSEEK_API_KEY",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": ""
        },
        "cloud": {
            "model": "gpt-4.1-mini",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "api_key": ""
        },
        "local": {
            "model": "qwen2.5:latest",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama"
        },
        "max_steps": 6,
        "timeout": 90,
        "speak_responses": True
    }
    try:
        user = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        for key, value in user.items():
            if isinstance(value, dict) and isinstance(defaults.get(key), dict):
                defaults[key].update(value)
            else:
                defaults[key] = value
    except (OSError, ValueError):
        pass
    return defaults


def _detect_ollama_model(base_url, preferred="qwen2.5:latest"):
    """Находит реально установленную модель в Ollama, если указанная отсутствует."""
    try:
        host = base_url.rstrip("/").removesuffix("/v1")
        req = urllib.request.Request(host + "/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name") for m in data.get("models", []) if m.get("name")]
            if not models:
                return preferred
            if preferred in models:
                return preferred
            pref_prefix = preferred.split(":")[0].lower()
            for m in models:
                if m.lower().startswith(pref_prefix):
                    return m
            return models[0]
    except Exception:
        return preferred


TOOLS = [
    {"type": "function", "function": {"name": "launch_app",
     "description": "Запустить приложение Windows по его названию, например Блокнот, калькулятор, Telegram или VS Code.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                    "required": ["name"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "set_volume",
     "description": "Установить системную громкость Windows от 0 до 100 процентов.",
     "parameters": {"type": "object", "properties": {"percent": {"type": "integer", "minimum": 0, "maximum": 100}},
                    "required": ["percent"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "set_brightness",
     "description": "Установить яркость встроенного экрана ноутбука от 0 до 100 процентов.",
     "parameters": {"type": "object", "properties": {"percent": {"type": "integer", "minimum": 0, "maximum": 100}},
                    "required": ["percent"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "browser_search",
     "description": "Открыть поиск Google в браузере по запросу.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "open_website",
     "description": "Открыть сайт в браузере. Передавай обычный домен или URL.",
     "parameters": {"type": "object", "properties": {"url": {"type": "string"}},
                    "required": ["url"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "type_text",
     "description": "Напечатать переданный пользователем текст в активном окне. Не использовать для паролей и секретов.",
     "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                    "required": ["text"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "active_window",
     "description": "Узнать заголовок активного окна Windows.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "media_control",
     "description": "Управлять Яндекс Музыкой: play, pause, next, prev, what, restart.",
     "parameters": {"type": "object", "properties": {
         "action": {"type": "string", "enum": ["play", "pause", "next", "prev", "what", "restart"]}},
         "required": ["action"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "set_dpi",
     "description": "Установить чувствительность мыши жестов руки (DPI от 400 до 2500) в Hand Control.",
     "parameters": {"type": "object", "properties": {"dpi": {"type": "integer", "minimum": 400, "maximum": 2500}},
                    "required": ["dpi"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "toggle_gestures",
     "description": "Включить (true) или выключить/поставить на паузу (false) управление мышью жестами руки.",
     "parameters": {"type": "object", "properties": {"enabled": {"type": "boolean"}},
                    "required": ["enabled"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "repick_hand",
     "description": "Сбросить текущую руку и заново выбрать управляющую руку перед камерой.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_hand_control_status",
     "description": "Узнать текущее состояние Hand Control X: DPI, пауза жестов, активное окно Windows.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "create_folder",
     "description": "Создать папку. Путь можно дать как ~/Desktop/имя, ~/Documents/имя или просто имя (создастся на рабочем столе). Промежуточные папки создаются автоматически. Можно создавать несколько папок за несколько вызовов.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "open_folder",
     "description": "Открыть папку в проводнике Windows по пути (~/Desktop, ~/Documents и т.п.).",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "list_folder",
     "description": "Показать содержимое папки: до 50 файлов и папок с пометкой [папка].",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "create_text_file",
     "description": "Создать текстовый файл с содержимым, например заметку или скрипт. Кодировка UTF-8.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "content": {"type": "string"}},
         "required": ["path"], "additionalProperties": False}}},
]


class AIAssistant:
    def __init__(self, feedback=None, speak=None, executor=None):
        self.settings = _load_settings()
        self.feedback = feedback or (lambda message, ok=True: print(message))
        self.speak = speak
        self.executor = executor
        self.history = []
        self.lock = threading.Lock()

    BLOCKED_PARTS = {"windows", "program files", "program files (x86)",
                     "$recycle.bin", "appdata", "system volume information"}

    @classmethod
    def _safe_path(cls, raw):
        """Путь только внутри профиля пользователя. Относительное имя -> рабочий стол."""
        if not raw or len(raw) > 300:
            raise ValueError("Пустой или слишком длинный путь")
        p = Path(os.path.expanduser(str(raw).strip().strip('"').strip("'")))
        if not p.is_absolute():
            p = Path.home() / "Desktop" / p
        p = p.resolve()
        home = Path.home().resolve()
        if home not in p.parents and p != home:
            raise ValueError("Разрешены только папки внутри профиля пользователя")
        if {part.lower() for part in p.parts} & cls.BLOCKED_PARTS:
            raise ValueError("Эта системная папка недоступна")
        return p

    def _provider_configs(self):
        mode = self.settings.get("provider", "auto").lower()
        deepseek = self.settings.get("deepseek", {})
        cloud = self.settings.get("cloud", {})
        local = self.settings.get("local", {})
        timeout = int(self.settings.get("timeout", TIMEOUT_DEFAULT))
        choices = []

        def get_key(cfg, default_env):
            env_name = cfg.get("api_key_env", default_env)
            return os.environ.get(env_name, "") or cfg.get("api_key", "")

        ds_key = get_key(deepseek, "DEEPSEEK_API_KEY")
        cloud_key = get_key(cloud, "OPENAI_API_KEY")

        if mode == "deepseek":
            if not ds_key:
                raise RuntimeError("Не задан ключ DeepSeek. Укажи его в ai_settings.json (deepseek.api_key) или переменной DEEPSEEK_API_KEY.")
            choices.append(("deepseek", deepseek.get("base_url", "https://api.deepseek.com/v1"),
                            deepseek.get("model", "deepseek-chat"), ds_key, timeout))
        elif mode == "cloud":
            if not cloud_key:
                raise RuntimeError("Не задан API-ключ OpenAI. Установи переменную OPENAI_API_KEY или укажи в ai_settings.json.")
            choices.append(("cloud", cloud.get("base_url", "https://api.openai.com/v1"),
                            cloud.get("model", "gpt-4.1-mini"), cloud_key, timeout))
        elif mode == "local":
            local_base = local.get("base_url", "http://localhost:11434/v1")
            raw_model = local.get("model", "qwen2.5:latest")
            actual_model = _detect_ollama_model(local_base, raw_model)
            choices.append(("local", local_base, actual_model, local.get("api_key", "ollama"), timeout))
        elif mode == "auto":
            # В авто-режиме: DeepSeek (если задан ключ) -> OpenAI (если задан) -> локальная Ollama
            if ds_key:
                choices.append(("deepseek", deepseek.get("base_url", "https://api.deepseek.com/v1"),
                                deepseek.get("model", "deepseek-chat"), ds_key, timeout))
            if cloud_key:
                choices.append(("cloud", cloud.get("base_url", "https://api.openai.com/v1"),
                                cloud.get("model", "gpt-4.1-mini"), cloud_key, timeout))
            local_base = local.get("base_url", "http://localhost:11434/v1")
            raw_model = local.get("model", "qwen2.5:latest")
            actual_model = _detect_ollama_model(local_base, raw_model)
            choices.append(("local", local_base, actual_model, local.get("api_key", "ollama"), timeout))
        else:
            raise RuntimeError("provider должен быть auto, deepseek, cloud или local")
        return choices

    @staticmethod
    def _request(base_url, model, api_key, messages, timeout=90):
        payload = json.dumps({
            "model": model, "messages": messages, "tools": TOOLS,
            "tool_choice": "auto", "temperature": 0.2
        }).encode("utf-8")
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions", data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:1000]
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Не удалось связаться с LLM: {exc}") from exc

    def _run_tool(self, name, args):
        # Все инструменты — allowlist. Произвольный Python или shell от модели не запускается.
        if name == "launch_app":
            app = str(args["name"]).strip()[:100]
            if not app:
                raise ValueError("Не указано приложение")
            import voice_windows
            result = voice_windows.act_launch(app, None)
            return str(result or f"Запускаю {app}")
        if name == "set_volume":
            value = int(args["percent"])
            if not 0 <= value <= 100:
                raise ValueError("Громкость должна быть от 0 до 100")
            import voice_windows
            voice_windows.set_volume(value)
            return f"Громкость установлена на {value}%"
        if name == "set_brightness":
            value = int(args["percent"])
            if not 0 <= value <= 100:
                raise ValueError("Яркость должна быть от 0 до 100")
            import voice_windows
            current = voice_windows.get_brightness()
            if current is None:
                return "Яркость доступна только на встроенном экране ноутбука"
            voice_windows.set_brightness(value)
            return f"Яркость установлена на {value}%"
        if name == "browser_search":
            query = str(args["query"]).strip()[:500]
            if not query:
                raise ValueError("Пустой поисковый запрос")
            webbrowser.open("https://www.google.com/search?q=" + quote_plus(query))
            return f"Открыл поиск: {query}"
        if name == "open_website":
            raw = str(args["url"]).strip()[:500]
            candidate = raw if "://" in raw else "https://" + raw
            parsed = urlparse(candidate)
            if parsed.scheme not in ("http", "https") or not parsed.netloc or " " in candidate:
                raise ValueError("Разрешены только корректные HTTP/HTTPS адреса")
            webbrowser.open(candidate)
            return f"Открыл сайт {candidate}"
        if name == "type_text":
            value = str(args["text"])[:3000]
            import pyperclip
            import pyautogui
            pyperclip.copy(value)
            pyautogui.hotkey("ctrl", "v")
            return f"Вставил текст ({len(value)} символов)"
        if name == "active_window":
            try:
                import pygetwindow
                window = pygetwindow.getActiveWindow()
                return "Активное окно: " + (window.title if window and hasattr(window, "title") else "не определено")
            except Exception:
                return "Активное окно не определено"
        if name == "media_control":
            action = str(args["action"])
            from yandex_music import YandexMusic
            ok, message = YandexMusic().command(action)
            if not ok:
                raise RuntimeError(message)
            return message
        if name == "set_dpi":
            dpi = int(args["dpi"])
            if self.executor and getattr(self.executor, "controller", None):
                self.executor.controller.set_dpi(dpi)
                return f"DPI установлен на {dpi}"
            return f"DPI {dpi} сохранён"
        if name == "toggle_gestures":
            enabled = bool(args["enabled"])
            if self.executor and getattr(self.executor, "controller", None):
                self.executor.controller.paused = not enabled
                self.executor.controller.release()
                return "Жесты включены" if enabled else "Жесты выключены (пауза)"
            return "Контроллер жестов не подключён"
        if name == "repick_hand":
            if self.executor and getattr(self.executor, "controller", None):
                self.executor.controller.relock()
                return "Рука сброшена: покажите руку заново"
            return "Контроллер жестов не подключён"
        if name == "get_hand_control_status":
            parts = []
            if self.executor and getattr(self.executor, "controller", None):
                ctrl = self.executor.controller
                parts.append(f"DPI: {ctrl.dpi}")
                parts.append(f"Жесты: {'пауза' if ctrl.paused else 'активны'}")
            else:
                parts.append("Жесты: не запущены")
            try:
                import pygetwindow
                win = pygetwindow.getActiveWindow()
                if win and hasattr(win, "title") and win.title:
                    parts.append(f"Окно: {win.title}")
            except Exception:
                pass
            return "; ".join(parts)
        if name in ("create_folder", "open_folder", "list_folder", "create_text_file"):
            path = self._safe_path(args.get("path"))
            if name == "create_folder":
                path.mkdir(parents=True, exist_ok=True)
                return f"Папка создана: {path}"
            if name == "open_folder":
                if not path.is_dir():
                    raise ValueError(f"Папки нет: {path}")
                os.startfile(str(path))
                return f"Открываю папку: {path}"
            if name == "list_folder":
                if not path.is_dir():
                    raise ValueError(f"Папки нет: {path}")
                items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
                if not items:
                    return f"Папка пуста: {path}"
                lines = [f"[папка] {x.name}" if x.is_dir() else x.name for x in items[:50]]
                more = f"\n... и ещё {len(items) - 50}" if len(items) > 50 else ""
                return f"{path}:\n" + "\n".join(lines) + more
            content = str(args.get("content") or "")[:20000]
            if path.suffix.lower() not in (".txt", ".md", ".csv", ".json", ".py", ".log"):
                path = path.with_suffix(".txt")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return f"Файл создан: {path} ({len(content)} символов)"
        raise ValueError(f"Инструмент не разрешён: {name}")

    def ask(self, user_text):
        if not self.settings.get("enabled", True):
            return "AI отключён в настройках."
        with self.lock:
            messages = [{
                "role": "system",
                "content": (
                    "Ты голосовой и текстовый ассистент проекта Hand Control X для Windows. "
                    "Ты умеешь общаться с пользователем, отвечать на любые вопросы и помогать в управлении компьютером и жестами. "
                    "Отвечай по-русски, кратко, дружелюбно и по делу (обычно 1-3 предложения), так как твои ответы могут озвучиваться и выводиться на экран камеры. "
                    "Если пользователь просит выполнить действие (изменить DPI, звук, открыть сайт, запустить программу, управлять жестами, создать папку/файл), используй подходящий инструмент. "
                    "Если пользователь просто общается, шутит или задает вопрос — отвечай естественно и поддерживай живой диалог. "
                    "Не придумывай результаты вызовов инструментов. Не выполняй опасные действия (удаление файлов, выключение ПК)."
                )
            }]
            messages.extend(self.history[-12:])
            messages.append({"role": "user", "content": str(user_text)[:2000]})
            errors = []
            for provider, base, model, key, timeout in self._provider_configs():
                try:
                    provider_messages = list(messages)
                    for _ in range(max(1, min(int(self.settings.get("max_steps", MAX_STEPS)), 8))):
                        data = self._request(base, model, key, provider_messages, timeout=timeout)
                        choice = data["choices"][0]["message"]
                        provider_messages.append(choice)
                        calls = choice.get("tool_calls") or []
                        if not calls:
                            answer = choice.get("content") or "Готово."
                            self.history.extend([
                                {"role": "user", "content": str(user_text)[:2000]},
                                {"role": "assistant", "content": answer}
                            ])
                            self.history = self.history[-20:]
                            return answer
                        for call in calls:
                            fn = call.get("function", {})
                            name = fn.get("name", "")
                            try:
                                args = json.loads(fn.get("arguments") or "{}")
                                result = self._run_tool(name, args)
                                tool_result = {"ok": True, "result": result}
                            except Exception as exc:
                                tool_result = {"ok": False, "error": str(exc)}
                            provider_messages.append({
                                "role": "tool", "tool_call_id": call["id"],
                                "content": json.dumps(tool_result, ensure_ascii=False)
                            })
                    return "Остановил выполнение: достигнут лимит шагов. Уточни задачу."
                except Exception as exc:
                    errors.append(f"{provider} ({model}): {exc}")
                    if self.settings.get("provider") != "auto":
                        break
            raise RuntimeError("Не удалось выполнить запрос. " + " | ".join(errors))
