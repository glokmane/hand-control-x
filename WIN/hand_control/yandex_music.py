# Copyright (c) 2026 Hand Control X authors. All rights reserved.
# Proprietary and confidential — see LICENSE. Do not redistribute.
"""
Управление Яндекс Музыкой через медиа-пульт Windows (System Media Transport Controls).

Браузеры и приложение Яндекс Музыки регистрируют в Windows "медиа-сессию" - ту же, что видна
в всплывающем окне громкости. Через неё можно переключать треки, ставить на паузу и перематывать
на точное время, даже если вкладка в фоне, а браузер свёрнут: окна и клавиатура не нужны.

Как понимаем, что это именно Яндекс Музыка:
  * приложение Яндекс Музыки - по имени медиа-сессии;
  * вкладка в браузере - по заголовку окна ("... на Яндекс Музыке ..."), пока вкладка активна.
    Запоминаем, какой медиа-сессии она принадлежит, и узнаём её дальше, когда вкладка уходит в фон.
"""
import asyncio
import datetime
import re
import threading
import time
import winreg

import pygetwindow

YANDEX_MARKERS = ("яндекс музык", "yandex music", "music.yandex")
SEEK_DEFAULT = 15          # "перемотай вперёд" без числа - на столько секунд
POLL_EVERY = 1.0           # как часто обновлять состояние для экрана, с


def _manager():
    from winrt.windows.media.control import GlobalSystemMediaTransportControlsSessionManager
    return GlobalSystemMediaTransportControlsSessionManager


def session_exe(aumid):
    """Имя медиа-сессии Windows -> программа браузера ('zen.exe', 'chrome.exe'...)."""
    a = aumid.lower()
    for key, exe in (("msedge", "msedge.exe"), ("chrome", "chrome.exe"), ("opera", "opera.exe"),
                     ("brave", "brave.exe"), ("vivaldi", "vivaldi.exe"), ("yandex", "browser.exe")):
        if key in a:
            return exe
    if a.endswith(".exe"):
        return a.rsplit("\\", 1)[-1]
    if re.fullmatch(r"[0-9a-f]{16}", a):
        # Firefox и браузеры на нём (Zen, LibreWolf...) называют сессию хэшем установки;
        # по нему в реестре находится, какой это exe
        for prog in ("FirefoxURL-", "FirefoxHTML-", "FirefoxPDF-"):
            try:
                with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, prog + aumid.upper() + r"\shell\open\command") as k:
                    cmd = winreg.QueryValue(k, None)
                return re.search(r"([^\\\"]+\.exe)", cmd, re.I).group(1).lower()
            except (OSError, AttributeError):
                continue
    return ""


def yandex_tab_browsers():
    """Программы браузеров, в которых сейчас активна вкладка Яндекс Музыки."""
    from voice_browser import is_browser, window_exe   # здесь, чтобы не было кругового импорта
    found = set()
    for w in pygetwindow.getAllWindows():
        if w.title and any(m in w.title.lower() for m in YANDEX_MARKERS) and is_browser(w._hWnd, w.title):
            found.add(window_exe(w._hWnd))
    return found


def is_yandex_app(aumid):
    a = aumid.lower()
    return "yandex" in a and "music" in a


class YandexMusic:
    """Находит медиа-сессию Яндекс Музыки и управляет ею. Состояние для экрана - в self.state."""

    def __init__(self):
        self.known = None                 # имя сессии, в которой однажды видели вкладку Яндекс Музыки
        self.state = None                 # {"where", "title", "artist", "playing", "pos", "dur"} или None
        self.error = None
        self._watcher = None

    # --- поиск ---
    async def _find(self):
        sessions = list((await _manager().request_async()).get_sessions())
        for s in sessions:
            if is_yandex_app(s.source_app_user_model_id):
                return s, "приложение"
        tabs = yandex_tab_browsers()
        for s in sessions:
            if tabs and session_exe(s.source_app_user_model_id) in tabs:
                self.known = s.source_app_user_model_id
                return s, "вкладка"
        for s in sessions:
            if s.source_app_user_model_id == self.known:
                return s, "вкладка в фоне"
        if self.known and not any(s.source_app_user_model_id == self.known for s in sessions):
            self.known = None             # вкладку закрыли - сессия пропала
        return None, None

    @staticmethod
    def _position(s):
        tl, pb = s.get_timeline_properties(), s.get_playback_info()
        pos = tl.position.total_seconds()
        if pb.playback_status == 4:       # играет: позиция обновляется не каждый миг - досчитываем
            pos += (datetime.datetime.now(datetime.timezone.utc) - tl.last_updated_time).total_seconds()
        return max(0.0, pos), tl.end_time.total_seconds(), pb.playback_status == 4

    async def _read(self):
        s, where = await self._find()
        if s is None:
            return None
        info = await s.try_get_media_properties_async()
        pos, dur, playing = self._position(s)
        return {"where": where, "title": info.title, "artist": info.artist, "playing": playing, "pos": pos, "dur": dur}

    def refresh(self):
        try:
            self.state, self.error = asyncio.run(self._read()), None
        except ImportError:
            self.state, self.error = None, "нет модуля winrt (pip install winrt-Windows.Media.Control)"
        except Exception as e:
            self.state, self.error = None, str(e)
        return self.state

    def watch(self):
        """Фоновое обновление состояния раз в секунду - для строки на экране."""
        if self._watcher is None:
            def loop():
                while True:
                    self.refresh()
                    time.sleep(POLL_EVERY)
            self._watcher = threading.Thread(target=loop, daemon=True)
            self._watcher.start()

    # --- команды: возвращают (успех, текст для экрана) ---
    def command(self, action, arg=None):
        try:
            return asyncio.run(self._command(action, arg))
        except ImportError:
            return False, "нет модуля winrt: pip install winrt-Windows.Media.Control"

    async def _command(self, action, arg):
        s, where = await self._find()
        if s is None:
            return False, "Яндекс Музыка не открыта - скажите «открой яндекс музыку»"
        info = await s.try_get_media_properties_async()
        track = f"{info.title} - {info.artist}" if info.artist else info.title
        if action == "what":
            pos, dur, playing = self._position(s)
            return True, f"{'Играет' if playing else 'На паузе'}: {track} ({fmt(pos)} из {fmt(dur)})"
        if action == "next":
            return await s.try_skip_next_async(), "следующий трек"
        if action == "prev":
            return await s.try_skip_previous_async(), "предыдущий трек"
        if action == "pause":
            return await s.try_pause_async(), f"пауза: {track}"
        if action == "play":
            return await s.try_play_async(), f"играет: {track}"
        if action in ("seek", "restart"):
            if not s.get_playback_info().controls.is_playback_position_enabled:
                return False, "этот плеер не даёт перематывать"
            pos, dur, _ = self._position(s)
            target = 0.0 if action == "restart" else min(max(0.0, pos + arg), max(0.0, dur - 1))
            ok = await s.try_change_playback_position_async(int(target * 10_000_000))
            if action == "restart":
                return ok, "трек сначала"
            return ok, f"{'вперёд' if arg > 0 else 'назад'} на {abs(arg):.0f} с -> {fmt(target)}"
        return False, f"неизвестное действие {action}"


def fmt(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"
