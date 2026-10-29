# Copyright (c) 2026 Hand Control X authors. All rights reserved.
# Proprietary and confidential — see LICENSE. Do not redistribute.
"""
Голосовое управление самой Windows: программы, окна, папки, громкость, яркость, питание.
Подключается к голосовому движку voice_browser.py - работает и в hand_control.py, и отдельно.

  программы:  "запусти телеграм" (любая установленная программа через поиск Windows),
              "открой проводник / блокнот / калькулятор / параметры / диспетчер задач / терминал"
  папки:      "открой папку загрузки / документы / рабочий стол / изображения"
  параметры:  "настройки звука / экрана / блютуз / вай фай / батареи", "обновления windows"
  окна:       "сверни окно", "разверни окно", "закрой программу", "окно влево / вправо",
              "сверни все" (рабочий стол), "следующее окно", "все окна", "переключись на телеграм"
  столы:      "новый рабочий стол", "следующий / предыдущий рабочий стол"
  звук:       "громкость 40", "громкость на максимум"     (громче / тише / без звука - в voice_browser)
  яркость:    "яркость 70", "ярче", "темнее"
  ответы:     "который час", "какое сегодня число", "заряд батареи" - компьютер отвечает голосом
  разное:     "скриншот области", "буфер обмена", "смени язык", "уведомления", "найди на компьютере ..."
  питание:    "заблокируй компьютер", "спящий режим",
              "выключи компьютер" / "перезагрузи компьютер" / "выйди из системы" -> переспросит, скажите "да";
              выключение и перезагрузка - через минуту, "отмени выключение" отменяет
"""
import ctypes
import ctypes.wintypes
import datetime
import os
import subprocess
import time
from pathlib import Path

import pyautogui
import pygetwindow

import voice_browser as vb     # общие помощники; этот модуль подключается из конца voice_browser.py

CREATE_NO_WINDOW = 0x08000000
SHUTDOWN_DELAY = 60            # выключение/перезагрузка - через столько секунд, можно отменить
CONFIRM_SECS = 8               # сколько ждать "да" для опасных команд

# встроенные программы Windows: открываются и словом "открой"
APPS = {
    "проводник": "explorer", "блокнот": "notepad", "калькулятор": "calc", "параметры": "ms-settings:",
    "настройки": "ms-settings:", "диспетчер задач": "taskmgr", "панель управления": "control",
    "командная строка": "cmd", "консоль": "cmd", "терминал": "wt", "пэйнт": "mspaint", "paint": "mspaint",
    "ножницы": "snippingtool", "этот компьютер": "shell:MyComputerFolder", "корзина": "shell:RecycleBinFolder",
}
# как программы называются в поиске Windows (русское название -> английское)
START_NAMES = {
    "телеграм": "Telegram", "телеграмм": "Telegram", "обсидиан": "Obsidian", "вс код": "Visual Studio Code",
    "визуал студио код": "Visual Studio Code", "ворд": "Word", "эксель": "Excel", "пауэрпоинт": "PowerPoint",
    "стим": "Steam", "дискорд": "Discord", "спотифай": "Spotify", "зум": "Zoom", "фотошоп": "Photoshop",
    "хром": "Chrome", "зен": "Zen", "яндекс браузер": "Yandex", "пайчарм": "PyCharm", "юпитер": "Jupyter",
    "аутлук": "Outlook", "скайп": "Skype", "вотсап": "WhatsApp", "ватсап": "WhatsApp",
}
FOLDERS = {
    "загрузки": "Downloads", "документы": "Documents", "рабочий стол": "Desktop", "изображения": "Pictures",
    "картинки": "Pictures", "музыка": "Music", "видео": "Videos",
}
SETTINGS = {
    "настройки звука": "ms-settings:sound", "настройки экрана": "ms-settings:display",
    "настройки блютуз": "ms-settings:bluetooth", "настройки вай фай": "ms-settings:network-wifi",
    "настройки сети": "ms-settings:network", "настройки батареи": "ms-settings:batterysaver",
    "электропитание": "ms-settings:powersleep", "обновления windows": "ms-settings:windowsupdate",
    "обновления": "ms-settings:windowsupdate", "настройки мыши": "ms-settings:mousetouchpad",
    "приложения": "ms-settings:appsfeatures", "настройки языка": "ms-settings:regionlanguage",
}
YES = ["да", "подтверждаю", "выполняй", "да выключи", "да перезагрузи", "конечно"]


# ---------------- окна ----------------

def target_window():
    """Окно, с которым работаем: активное, а если активно окно камеры - то, что под ним."""
    try:
        active = pygetwindow.getActiveWindow()
    except Exception:
        active = None
    if active and active.title and not any(o in active.title for o in vb.OWN_WINDOWS):
        return active
    for w in pygetwindow.getAllWindows():
        if (w.title and w.visible and w.width > 200 and not w.isMinimized
                and not any(o in w.title for o in vb.OWN_WINDOWS) and w.title not in ("Program Manager",)):
            return w
    return None


def with_target(keys=None, method=None):
    """Действие над окном: выводим его вперёд, потом жмём клавиши или вызываем метод окна."""
    def act(arg, ex):
        w = target_window()
        if w is None:
            return "нет подходящего окна"
        vb.bring_to_front(w._hWnd)
        if method:
            try:
                getattr(w, method)()
            except Exception:
                pass
        if keys:
            pyautogui.hotkey(*keys)
        return ("ok", f"{ACT_NAMES.get(method or '+'.join(keys), '')}: {w.title[:40]}")
    return act


ACT_NAMES = {"minimize": "свернул", "maximize": "развернул", "restore": "восстановил", "alt+f4": "закрыл",
             "winleft+left": "влево", "winleft+right": "вправо"}


def act_switch_to(arg, ex):
    """'переключись на телеграм' - находим окно по названию программы или заголовку."""
    name = START_NAMES.get(arg, arg).lower()
    for w in pygetwindow.getAllWindows():
        if not (w.title and w.visible) or any(o in w.title for o in vb.OWN_WINDOWS):
            continue
        if name in w.title.lower() or name.replace(" ", "") in vb.window_exe(w._hWnd).replace(".exe", ""):
            vb.bring_to_front(w._hWnd)
            return ("ok", f"переключился: {w.title[:40]}")
    return f"не нашёл открытое окно «{arg}»"


# ---------------- программы, папки, параметры ----------------

def act_app(arg, ex):
    os.startfile(APPS[arg])
    return ("ok", f"открываю {arg}")


def act_launch(arg, ex):
    """Любая установленная программа: Win, печатаем название в поиске, Enter."""
    if arg in APPS:
        return act_app(arg, ex)
    name = START_NAMES.get(arg, arg)
    pyautogui.press("winleft")
    time.sleep(0.7)
    vb.paste(name)
    time.sleep(0.9)
    pyautogui.press("enter")
    return ("ok", f"запускаю {name}")


def act_folder(arg, ex):
    key = next((k for k in FOLDERS if vb.words_match(arg.split(), k.split())), None)
    if key is None:
        return f"не знаю папку «{arg}»"
    path = Path.home() / FOLDERS[key]
    os.startfile(str(path if path.exists() else Path.home()))
    return ("ok", f"папка {key}")


def act_settings(page):
    def act(arg, ex):
        os.startfile(page)
    return act


def act_search_pc(arg, ex):
    pyautogui.hotkey("winleft", "s")
    time.sleep(0.7)
    vb.paste(arg)
    return ("ok", f"поиск на компьютере: {arg}")


# ---------------- звук и яркость ----------------

def _endpoint_volume():
    import comtypes
    from pycaw.pycaw import AudioUtilities
    comtypes.CoInitialize()        # команды выполняются в своём потоке - COM нужно включить в нём
    dev = AudioUtilities.GetSpeakers()
    if hasattr(dev, "EndpointVolume"):
        return dev.EndpointVolume
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import IAudioEndpointVolume
    return ctypes.cast(dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None),
                       ctypes.POINTER(IAudioEndpointVolume))


def get_volume():
    return round(_endpoint_volume().GetMasterVolumeLevelScalar() * 100)


def set_volume(percent):
    vol = _endpoint_volume()
    vol.SetMute(0, None)
    vol.SetMasterVolumeLevelScalar(max(0, min(100, percent)) / 100, None)


def act_volume(arg, ex):
    try:
        set_volume(arg)
    except ImportError:
        return "нет модуля pycaw: pip install pycaw"
    return ("ok", f"громкость {arg}%")


def _powershell(cmd):
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                          timeout=15, creationflags=CREATE_NO_WINDOW).stdout.strip()


def get_brightness():
    out = _powershell("(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness).CurrentBrightness")
    return int(out.split()[0]) if out else None


def set_brightness(percent):
    _powershell("Invoke-CimMethod -InputObject (Get-CimInstance -Namespace root/WMI -ClassName "
                "WmiMonitorBrightnessMethods) -MethodName WmiSetBrightness "
                f"-Arguments @{{Timeout=1; Brightness={int(max(0, min(100, percent)))}}}")


def act_brightness(arg, ex):
    current = get_brightness()
    if current is None:
        return "яркость меняется только у встроенного экрана ноутбука"
    target = current + arg[1] if arg[0] == "delta" else arg[1]
    target = max(0, min(100, target))
    set_brightness(target)
    return ("ok", f"яркость {target}%")


# ---------------- ответы голосом ----------------

class _PowerStatus(ctypes.Structure):
    _fields_ = [("ac", ctypes.c_byte), ("flag", ctypes.c_byte), ("percent", ctypes.c_byte),
                ("saver", ctypes.c_byte), ("life", ctypes.wintypes.DWORD), ("full", ctypes.wintypes.DWORD)]


def battery():
    s = _PowerStatus()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
    return (None if s.percent in (-1, 255) else s.percent & 0xFF), s.ac == 1


MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября",
          "ноября", "декабря"]
WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]


def act_time(arg, ex):
    now = datetime.datetime.now()
    return ("say", f"Сейчас {now.hour}:{now.minute:02d}")


def act_date(arg, ex):
    d = datetime.date.today()
    return ("say", f"Сегодня {WEEKDAYS[d.weekday()]}, {d.day} {MONTHS[d.month - 1]}")


def act_battery(arg, ex):
    percent, plugged = battery()
    if percent is None:
        return ("say", "Батарея не найдена")
    return ("say", f"Заряд {percent} процентов" + (", заряжается" if plugged else ""))


# ---------------- питание ----------------

def act_lock(arg, ex):
    ctypes.windll.user32.LockWorkStation()


def act_sleep(arg, ex):
    def go():
        time.sleep(2)              # успеть показать ответ
        ctypes.windll.powrprof.SetSuspendState(False, True, False)
    import threading
    threading.Thread(target=go, daemon=True).start()
    return ("ok", "спящий режим через 2 секунды")


def _shutdown(flag, text):
    def run():
        subprocess.run(["shutdown", flag, "/t", str(SHUTDOWN_DELAY)], creationflags=CREATE_NO_WINDOW)
        return ("say", f"{text} через {SHUTDOWN_DELAY} секунд. Скажите «отмени выключение», чтобы отменить")
    return run


def confirm(question, run):
    """Опасная команда: сначала спрашиваем, выполняем только после "да"."""
    def act(arg, ex):
        ex.pending_confirm = (run, time.time() + CONFIRM_SECS)
        return ("say", f"{question} Скажите «да» для подтверждения")
    return act


def act_logout_now():
    subprocess.run(["shutdown", "/l"], creationflags=CREATE_NO_WINDOW)
    return ("ok", "выход из системы")


def act_cancel_shutdown(arg, ex):
    subprocess.run(["shutdown", "/a"], creationflags=CREATE_NO_WINDOW)
    return ("say", "Выключение отменено")


# ---------------- таблицы команд (подмешиваются в voice_browser) ----------------

SYSTEM_FIXED = [
    (("сверни окно", "сверни это окно"), "свернуть окно", None, with_target(method="minimize")),
    (("разверни окно", "разверни на весь экран", "окно на весь экран"), "развернуть окно", None,
     with_target(method="maximize")),
    (("восстанови окно", "обычное окно"), "восстановить окно", None, with_target(method="restore")),
    (("закрой программу", "закрой приложение", "закрой это окно"), "закрыть программу", None,
     with_target(keys=("alt", "f4"))),
    (("окно влево", "прижми влево"), "окно влево", None, with_target(keys=("winleft", "left"))),
    (("окно вправо", "прижми вправо"), "окно вправо", None, with_target(keys=("winleft", "right"))),
    (("сверни все", "свернуть все", "покажи рабочий стол"), "рабочий стол", None, vb.hotkey("winleft", "d")),
    (("следующее окно", "переключи окно", "другое окно"), "следующее окно", None, vb.hotkey("alt", "tab")),
    (("все окна", "покажи все окна"), "все окна", None, vb.hotkey("winleft", "tab")),
    (("новый рабочий стол",), "новый рабочий стол", None, vb.hotkey("winleft", "ctrl", "d")),
    (("следующий рабочий стол",), "следующий рабочий стол", None, vb.hotkey("winleft", "ctrl", "right")),
    (("предыдущий рабочий стол",), "предыдущий рабочий стол", None, vb.hotkey("winleft", "ctrl", "left")),
    (("закрой рабочий стол",), "закрыть рабочий стол", None, vb.hotkey("winleft", "ctrl", "f4")),
    (("скриншот области", "вырезать экран", "снимок области"), "выделите область", None,
     vb.hotkey("winleft", "shift", "s")),
    (("буфер обмена", "история буфера"), "буфер обмена", None, vb.hotkey("winleft", "v")),
    (("смени язык", "переключи язык", "другой язык"), "язык переключен", None, vb.hotkey("winleft", "space")),
    (("уведомления", "покажи уведомления"), "уведомления", None, vb.hotkey("winleft", "n")),
    (("быстрые настройки", "центр управления"), "быстрые настройки", None, vb.hotkey("winleft", "a")),
    (("пуск", "меню пуск"), "пуск", None, vb.press("winleft")),
    (("громкость на максимум", "максимальная громкость"), "громкость 100%", None, lambda a, ex: act_volume(100, ex)),
    (("ярче", "сделай ярче", "прибавь яркость"), "ярче", None, lambda a, ex: act_brightness(("delta", 10), ex)),
    (("темнее", "сделай темнее", "убавь яркость"), "темнее", None, lambda a, ex: act_brightness(("delta", -10), ex)),
    (("который час", "сколько времени", "время"), "время", None, act_time),
    (("какое сегодня число", "какая дата", "какое число", "какой сегодня день"), "дата", None, act_date),
    (("заряд батареи", "сколько заряда", "батарея", "заряд"), "батарея", None, act_battery),
    (("заблокируй компьютер", "заблокируй экран", "блокировка"), "блокировка", None, act_lock),
    (("спящий режим", "усыпи компьютер", "режим сна"), "сон", None, act_sleep),
    (("выключи компьютер", "выключение компьютера"), "выключение", None,
     confirm("Выключить компьютер?", _shutdown("/s", "Компьютер выключится"))),
    (("перезагрузи компьютер", "перезагрузка компьютера"), "перезагрузка", None,
     confirm("Перезагрузить компьютер?", _shutdown("/r", "Компьютер перезагрузится"))),
    (("выйди из системы", "выход из системы"), "выход из системы", None,
     confirm("Выйти из системы? Несохранённые данные пропадут.", act_logout_now)),
    (("отмени выключение", "отмена выключения", "не выключай"), "выключение отменено", None, act_cancel_shutdown),
] + [((name,), name, None, act_settings(page)) for name, page in SETTINGS.items()]

SYSTEM_PREFIXED = [
    (("найди на компьютере", "поиск на компьютере", "найди в windows"), "поиск на компьютере", None, act_search_pc),
    (("открой папку", "папка"), "папка", None, act_folder),
    (("запусти", "открой программу", "включи программу"), "запуск", None, act_launch),
    (("переключись на", "переключись в", "покажи окно"), "переключиться", None, act_switch_to),
]

SYSTEM_HELP = [
    "windows: запусти <программу> | открой проводник / блокнот / параметры | открой папку загрузки",
    "    сверни / разверни окно | закрой программу | переключись на <...> | сверни все",
    "    громкость 40 | яркость 70 | ярче | темнее | который час | заряд батареи",
    "    заблокируй компьютер | спящий режим | выключи компьютер (переспросит) | отмени выключение",
]


def system_request(words):
    """Команды с числом и встроенные программы -> (описание, куда, действие, аргумент) или None."""
    number = vb.parse_number(words)
    if words and vb.word_eq(words[0], "громкость") and len(words) >= 2:
        if number is not None and 0 <= number <= 100:
            return f"громкость {number}%", None, act_volume, number
    if words and vb.word_eq(words[0], "яркость") and len(words) >= 2:
        if number is not None and 0 <= number <= 100:
            return f"яркость {number}%", None, act_brightness, ("set", number)
    if len(words) >= 2 and vb.word_eq(words[0], "открой"):
        rest = words[1:]
        app = next((a for a in APPS if vb.words_match(rest, a.split())), None)
        if app:
            return f"открыть {app}", None, act_app, app
        page = next((p for p in SETTINGS if vb.words_match(rest, p.split())), None)
        if page:
            return page, None, act_settings(SETTINGS[page]), None
    return None
