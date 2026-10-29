# Copyright (c) 2026 Hand Control X authors. All rights reserved.
# Proprietary and confidential — see LICENSE. Do not redistribute.
"""
Голосовое управление браузером - офлайн, по-русски, через Vosk.

Запуск только голосом (без камеры):   python voice_browser.py
Вместе с управлением жестами:          python hand_control.py

Скажите "команды" - появится список. Основное:
  поиск и сайты:   "найди рецепт борща", "найди на ютубе котиков", "открой ютуб", "открой хабр точка ком",
                   "открой в новой вкладке википедию"
  ссылки:          "нажми войти" (ищет текст на странице и нажимает ссылку/кнопку),
                   "следующая ссылка" / "предыдущая ссылка" + "ввод"
  навигация:       "назад", "вперёд", "обнови", "стоп"
  вкладки:         "новая вкладка", "закрой вкладку", "верни вкладку", "следующая / предыдущая вкладка",
                   "вкладка три", "последняя вкладка"
  прокрутка:       "вниз", "вверх", "ниже", "выше", "в начало", "в конец"
  страница:        "увеличь", "уменьши", "обычный масштаб", "полный экран", "найди на странице ...",
                   "в закладки", "история", "загрузки", "адресная строка"
  текст:           "напечатай ...", "ввод", "стереть", "очисти поле", "выдели всё", "копировать", "вставить",
                   "диктовка" - всё сказанное печатается ("точка", "запятая", "новая строка"),
                   выход - "стоп диктовка"
  видео и звук:    "пауза" / "играй", "громче", "тише", "без звука"
  Яндекс Музыка:   "следующий трек", "предыдущий трек", "перемотай вперёд" (на 15 с), "перемотай назад
                   на 30 секунд", "вперёд на минуту", "трек сначала", "останови музыку", "включи музыку",
                   "что играет", "открой яндекс музыку". Работает и когда вкладка в фоне, а браузер свёрнут.
  прочее:          "открой браузер", "скриншот", "выключи жесты" / "включи жесты"
"""
import ctypes
import ctypes.wintypes
import difflib
import json
import os
import queue
import re
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

import pyautogui
import pygetwindow
import pyperclip

from yandex_music import SEEK_DEFAULT, YandexMusic, fmt

VOSK_MODEL_NAME = "vosk-model-small-ru-0.22"
VOSK_MODEL_URL = f"https://alphacephei.com/vosk/models/{VOSK_MODEL_NAME}.zip"
# Vosk не открывает модель по пути с кириллицей, поэтому она лежит в AppData, а не рядом со скриптом
VOSK_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "vosk"
VOICE_MIN_CONF = 0.5          # фразы, в которых распознаватель уверен меньше (шум), игнорируются

BROWSER_EXES = {"zen.exe", "firefox.exe", "chrome.exe", "msedge.exe", "browser.exe", "opera.exe", "brave.exe",
                "vivaldi.exe", "arc.exe", "librewolf.exe", "waterfox.exe", "floorp.exe", "chromium.exe",
                "thorium.exe"}   # browser.exe - Яндекс Браузер
BROWSER_TITLES = ("Chrome", "Edge", "Firefox", "Zen", "Opera", "Яндекс", "Yandex", "Brave", "Vivaldi", "Chromium")
OWN_WINDOWS = ("Hand Control",)   # окно камеры: сюда печатать не надо, переключаемся на браузер

SITES = {
    "ютуб": "youtube.com", "ютюб": "youtube.com", "гугл": "google.com", "яндекс": "ya.ru",
    "вконтакте": "vk.com", "вк": "vk.com", "контакт": "vk.com", "википедия": "ru.wikipedia.org",
    "википедию": "ru.wikipedia.org", "вики": "ru.wikipedia.org", "почта": "mail.google.com",
    "почту": "mail.google.com", "джимейл": "mail.google.com", "телеграм": "web.telegram.org",
    "телеграмм": "web.telegram.org", "переводчик": "translate.google.com", "карты": "yandex.ru/maps",
    "гитхаб": "github.com", "чат гпт": "chatgpt.com", "кинопоиск": "kinopoisk.ru", "озон": "ozon.ru",
    "вайлдберриз": "wildberries.ru", "авито": "avito.ru", "погода": "yandex.ru/pogoda",
    "яндекс музыка": "music.yandex.ru", "яндекс музыку": "music.yandex.ru", "музыка": "music.yandex.ru",
    "музыку": "music.yandex.ru",
}
# Музыка (Яндекс Музыка через медиа-пульт Windows, см. yandex_music.py)
MUSIC_PHRASES = {
    "next": ["следующий трек", "следующая песня", "следующую песню", "следующий", "переключи трек",
             "другой трек", "другую песню"],
    "prev": ["предыдущий трек", "предыдущая песня", "предыдущую песню", "прошлый трек", "прошлая песня",
             "прошлую песню", "верни трек"],
    "pause": ["останови музыку", "стоп музыка", "пауза музыка", "музыку на паузу", "поставь музыку на паузу",
              "выключи музыку", "останови трек"],
    "play": ["включи музыку", "играй музыку", "продолжи музыку", "продолжить музыку", "запусти музыку",
             "включи трек", "музыку дальше"],
    "restart": ["трек сначала", "трек заново", "заново трек", "сначала", "с начала"],
    "what": ["что играет", "что сейчас играет", "какой трек", "какая песня", "что за песня", "что за трек"],
}
SEEK_VERBS = ("перемотай", "перемотать", "промотай", "мотай", "отмотай", "перемотка")
NUMBERS = {
    "один": 1, "одна": 1, "первая": 1, "первую": 1, "два": 2, "две": 2, "вторая": 2, "вторую": 2,
    "три": 3, "третья": 3, "третью": 3, "четыре": 4, "четвертая": 4, "четвертую": 4,
    "пять": 5, "пятая": 5, "пятую": 5, "шесть": 6, "шестая": 6, "шестую": 6,
    "семь": 7, "седьмая": 7, "седьмую": 7, "восемь": 8, "восьмая": 8, "восьмую": 8,
}
TLD = {"ком": "com", "ру": "ru", "орг": "org", "нет": "net", "ио": "io", "дев": "dev", "рф": "рф"}
TRANSLIT = dict(zip("абвгдежзийклмнопрстуфхцчшщъыьэюя",
                    ["a", "b", "v", "g", "d", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s",
                     "t", "u", "f", "h", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"]))
PUNCTUATION = [("вопросительный знак", "?"), ("восклицательный знак", "!"), ("с новой строки", "\n"),
               ("новая строка", "\n"), ("точка с запятой", ";"), ("двоеточие", ":"), ("запятая", ","),
               ("точка", "."), ("тире", " -")]
STOP_DICTATION = ["стоп диктовка", "конец диктовки", "закончить диктовку", "выключи диктовку", "хватит"]

DPI_MIN, DPI_MAX = 400, 2500
DPI_WAIT = 12                  # сколько секунд ждать число после "изменить ди-пи-ай"
# "ди-пи-ай" маленькая модель слышит как "ди первой", "дтп и"... поэтому ловим глагол + слово на "д"
DPI_VERBS = ("изменить", "измени", "поменять", "поменяй", "сменить", "смени", "установить", "установи",
             "поставь", "задай", "задать")
NUM_WORDS = {
    "ноль": 0, "один": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6,
    "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
    "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18,
    "девятнадцать": 19, "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60,
    "семьдесят": 70, "восемьдесят": 80, "девяносто": 90, "сто": 100, "двести": 200, "триста": 300,
    "четыреста": 400, "пятьсот": 500, "шестьсот": 600, "семьсот": 700, "восемьсот": 800, "девятьсот": 900,
}
THOUSANDS = ("тысяча", "тысячи", "тысяч", "тысячу")
FILLERS = {"и", "а", "ну", "пожалуйста", "давай", "эй"}   # слова-паразиты, которые не мешают понять команду


def normalize(text):
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def ensure_vosk_model():
    path = VOSK_DIR / VOSK_MODEL_NAME
    if not path.exists():
        VOSK_DIR.mkdir(parents=True, exist_ok=True)
        print("Скачиваю русскую модель Vosk (~45 МБ)...")
        archive = VOSK_DIR / f"{VOSK_MODEL_NAME}.zip"
        urllib.request.urlretrieve(VOSK_MODEL_URL, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(VOSK_DIR)
        archive.unlink()
    return path


# ======================= распознавание речи =======================

class VoiceListener(threading.Thread):
    """Фоновый поток: микрофон -> Vosk -> законченные фразы в очереди self.texts."""

    def __init__(self):
        super().__init__(daemon=True)
        self.texts = queue.SimpleQueue()
        self.partial = ""            # то, что распознаётся прямо сейчас (для подсказки на экране)
        self.ignore_until = 0.0      # пока компьютер говорит сам, фразы с микрофона не принимаем
        self.status = "loading"
        self.running = True

    def run(self):
        try:  # голос - необязательная часть: без vosk или микрофона жесты продолжают работать
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except ImportError:
            self.status = "no vosk"
            return
        try:
            SetLogLevel(-1)
            rec = self.make_recognizer(Model(str(ensure_vosk_model())), KaldiRecognizer)
            audio = queue.Queue()
            with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype="int16", channels=1,
                                   callback=lambda data, *_: audio.put(bytes(data))):
                self.status = "on"
                while self.running:
                    try:
                        chunk = audio.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    text = self.process(chunk, rec)
                    if text:
                        self.texts.put(text)
        except Exception as e:
            self.status = "error"
            print("Голосовые команды отключены:", e)

    @staticmethod
    def make_recognizer(model, recognizer_cls):
        rec = recognizer_cls(model, 16000)
        rec.SetWords(True)
        return rec

    def process(self, chunk, rec):
        """Кусок звука -> законченная фраза или None."""
        if not rec.AcceptWaveform(chunk):
            self.partial = json.loads(rec.PartialResult()).get("partial", "")
            return None
        self.partial = ""
        result = json.loads(rec.Result())
        words = result.get("result", [])
        if not words or sum(w["conf"] for w in words) / len(words) < VOICE_MIN_CONF:
            return None
        if time.time() < self.ignore_until:   # это компьютер договаривал свой ответ - не команда
            return None
        return result.get("text") or None


# ======================= действия =======================

def hotkey(*keys):
    return lambda arg, ex: pyautogui.hotkey(*keys)


def press(key, times=1):
    return lambda arg, ex: pyautogui.press(key, presses=times, interval=0.03)


def paste(text):
    """Печать через буфер обмена: pyautogui.write не умеет кириллицу. Старый буфер возвращаем."""
    try:
        old = pyperclip.paste()
    except Exception:
        old = None
    pyperclip.copy(text)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.15)
    if old is not None:
        pyperclip.copy(old)


def go(address):
    """Адресная строка: адрес сайта откроется, любой другой текст уйдёт в поисковик браузера."""
    pyautogui.hotkey("ctrl", "l")
    time.sleep(0.1)
    paste(address)
    pyautogui.press("enter")


def site_address(name):
    name = name.removeprefix("сайт ").strip()
    if name in SITES:
        return SITES[name]
    for key, url in SITES.items():                 # "википедии" вместо "википедию"
        if words_match(name.split(), key.split()):
            return url
    if " точка " in name:                          # "хабр точка ком" -> habr.com
        *parts, tld = name.split(" точка ")
        host = ".".join("".join(TRANSLIT.get(c, c) for c in p.replace(" ", "")) for p in parts)
        return f"{host}.{TLD.get(tld, ''.join(TRANSLIT.get(c, c) for c in tld))}"
    return name                                    # неизвестный сайт - браузер поищет его сам


def act_open_site(arg, ex):
    go(site_address(arg))


def act_open_new_tab(arg, ex):
    pyautogui.hotkey("ctrl", "t")
    time.sleep(0.3)
    go(site_address(arg))


def act_search(arg, ex):
    go(arg)


def act_youtube(arg, ex):
    go("https://www.youtube.com/results?search_query=" + urllib.parse.quote(arg))


def act_find_on_page(arg, ex):
    pyautogui.hotkey("ctrl", "f")
    time.sleep(0.2)
    paste(arg)
    pyautogui.press("enter")


def act_click_text(arg, ex):
    """Найти текст на странице и нажать: после поиска и Esc браузер фокусирует найденную ссылку."""
    act_find_on_page(arg, ex)
    time.sleep(0.3)
    pyautogui.press("esc")
    time.sleep(0.1)
    pyautogui.press("enter")


def act_type(arg, ex):
    paste(arg)


def act_tab(arg, ex):
    pyautogui.hotkey("ctrl", str(arg))


def act_open_browser(arg, ex):
    webbrowser.open("https://www.google.com")


def act_screenshot(arg, ex):
    folder = Path.home() / "Pictures" / "Screenshots"
    folder.mkdir(parents=True, exist_ok=True)
    pyautogui.screenshot(str(folder / time.strftime("screenshot_%Y%m%d_%H%M%S.png")))


def act_gestures(on):
    def act(arg, ex):
        if ex.controller is None:
            return "жесты не запущены"
        ex.controller.paused = not on
        ex.controller.release()
    return act


def act_dictation(arg, ex):
    ex.dictation, ex.capital, ex.first_word = True, True, True


def act_dpi(arg, ex):
    if ex.controller is None:
        return "жесты не запущены"
    if arg is not None and DPI_MIN <= arg <= DPI_MAX:
        ex.controller.set_dpi(arg)
        return ("ok", f"DPI = {arg}")
    ex.await_dpi_until = time.time() + DPI_WAIT
    return ("ok", f"Назовите DPI от {DPI_MIN} до {DPI_MAX}")


def act_music(arg, ex):
    ok, text = ex.music.command(*arg)
    return ("ok", text) if ok else text


def act_relock(arg, ex):
    if ex.controller is None:
        return "жесты не запущены"
    ex.controller.relock()
    return ("ok", "покажите руку, которой будете управлять")


def act_help(arg, ex):
    ex.help_until = time.time() + 15
    print("\n".join(HELP_LINES))


def act_clear_field(arg, ex):
    pyautogui.hotkey("ctrl", "a")
    pyautogui.press("backspace")


# Команды-фразы: фразы, описание, куда отправлять клавиши, действие.
# Куда: "browser" - только в браузер (найдём и активируем его окно), "text" - в любое окно,
# кроме окна камеры, None - фокус не важен.
FIXED = [
    (("назад", "вернись", "вернуться назад"), "назад", "browser", hotkey("alt", "left")),
    (("вперед",), "вперёд", "browser", hotkey("alt", "right")),
    (("обнови", "обновить", "перезагрузи", "обнови страницу"), "обновить страницу", "browser", press("f5")),
    (("стоп", "отмена", "отменить", "закрой поиск"), "Esc", "browser", press("esc")),
    (("новая вкладка", "открой новую вкладку", "открой вкладку"), "новая вкладка", "browser", hotkey("ctrl", "t")),
    (("закрой вкладку", "закрыть вкладку"), "закрыть вкладку", "browser", hotkey("ctrl", "w")),
    (("верни вкладку", "восстанови вкладку"), "вернуть вкладку", "browser", hotkey("ctrl", "shift", "t")),
    (("следующая вкладка", "вкладка вправо"), "следующая вкладка", "browser", hotkey("ctrl", "tab")),
    (("предыдущая вкладка", "вкладка влево"), "предыдущая вкладка", "browser", hotkey("ctrl", "shift", "tab")),
    (("последняя вкладка",), "последняя вкладка", "browser", hotkey("ctrl", "9")),
    (("новое окно",), "новое окно", "browser", hotkey("ctrl", "n")),
    (("закрой окно", "закрой браузер"), "закрыть окно браузера", "browser", hotkey("ctrl", "shift", "w")),
    (("вниз", "листай вниз", "прокрути вниз"), "страница вниз", "browser", press("pagedown")),
    (("вверх", "листай вверх", "прокрути вверх"), "страница вверх", "browser", press("pageup")),
    (("ниже", "чуть ниже"), "чуть ниже", "browser", press("down", 5)),
    (("выше", "чуть выше"), "чуть выше", "browser", press("up", 5)),
    (("в начало", "наверх", "в самый верх", "в начало страницы"), "в начало страницы", "browser", press("home")),
    (("в конец", "в самый низ", "в конец страницы"), "в конец страницы", "browser", press("end")),
    (("увеличь", "увеличить", "крупнее", "приблизь"), "крупнее", "browser", hotkey("ctrl", "add")),
    (("уменьши", "уменьшить", "мельче", "отдали"), "мельче", "browser", hotkey("ctrl", "subtract")),
    (("обычный масштаб", "сбрось масштаб"), "масштаб 100%", "browser", hotkey("ctrl", "0")),
    (("полный экран", "на весь экран"), "полный экран", "browser", press("f11")),
    (("в закладки", "добавь в закладки", "закладка"), "в закладки", "browser", hotkey("ctrl", "d")),
    (("история",), "история", "browser", hotkey("ctrl", "h")),
    (("загрузки",), "загрузки", "browser", hotkey("ctrl", "j")),
    (("адресная строка", "адрес"), "адресная строка", "browser", hotkey("ctrl", "l")),
    (("следующая ссылка", "следующее поле", "следующая кнопка"), "следующая ссылка", "browser", press("tab")),
    (("предыдущая ссылка", "предыдущее поле", "предыдущая кнопка"), "предыдущая ссылка", "browser",
     hotkey("shift", "tab")),
    (("ввод", "энтер", "отправить", "отправь", "подтвердить", "перейти"), "Enter", "text", press("enter")),
    (("стереть", "сотри", "удали слово"), "стереть слово", "text", hotkey("ctrl", "backspace")),
    (("очисти поле", "очистить поле", "стереть все", "сотри все"), "очистить поле", "text", act_clear_field),
    (("выдели все", "выделить все"), "выделить всё", "text", hotkey("ctrl", "a")),
    (("копировать", "скопируй"), "копировать", "text", hotkey("ctrl", "c")),
    (("вставить", "вставь"), "вставить", "text", hotkey("ctrl", "v")),
    (("пауза", "плей", "играй", "воспроизвести", "продолжи видео", "стоп видео"), "пауза / воспроизведение",
     None, press("playpause")),
    (("громче", "сделай громче"), "громче", None, press("volumeup", 5)),
    (("тише", "сделай тише"), "тише", None, press("volumedown", 5)),
    (("без звука", "выключи звук", "включи звук"), "звук вкл/выкл", None, press("volumemute")),
    (("открой браузер", "браузер"), "открыть браузер", None, act_open_browser),
    (("скриншот", "снимок экрана"), "скриншот сохранён", None, act_screenshot),
    (("выключи жесты", "отключи жесты"), "жесты выключены", None, act_gestures(False)),
    (("включи жесты",), "жесты включены", None, act_gestures(True)),
    (("диктовка", "начать диктовку", "режим диктовки"), "диктовка: говорите текст", "text", act_dictation),
    (("сменить руку", "смени руку", "другая рука", "выбрать руку"), "сменить руку", None, act_relock),
    (("команды", "помощь", "что ты умеешь"), "список команд", None, act_help),
]
FIXED_INDEX = {phrase: cmd[1:] for cmd in FIXED for phrase in cmd[0]}

# Команды с продолжением ("найди <что>"). Порядок важен: длинные приставки раньше коротких.
PREFIXED = [
    (("найди на странице", "найти на странице", "поиск на странице"), "найти на странице", "browser",
     act_find_on_page),
    (("найди на ютубе", "найди в ютубе", "найди на ютюбе", "ютуб"), "поиск на YouTube", "browser", act_youtube),
    (("нажми", "кликни", "перейди по ссылке", "открой ссылку"), "нажать", "browser", act_click_text),
    (("найди", "найти", "поищи", "загугли", "погугли", "поиск"), "поиск", "browser", act_search),
    (("открой в новой вкладке",), "открыть в новой вкладке", "browser", act_open_new_tab),
    (("открой сайт", "зайди на", "перейди на", "открой", "зайди в"), "открыть", "browser", act_open_site),
    (("напечатай", "напиши", "введи"), "напечатать", "text", act_type),
]

def _site_url(arg):
    addr = site_address(arg)
    if "." in addr and " " not in addr:
        return "https://" + addr
    return "https://www.google.com/search?q=" + urllib.parse.quote(addr)


# Если окно браузера не удалось активировать, эти команды открываются прямой ссылкой
DIRECT_URL = {
    act_search: lambda arg: "https://www.google.com/search?q=" + urllib.parse.quote(arg),
    act_youtube: lambda arg: "https://www.youtube.com/results?search_query=" + urllib.parse.quote(arg),
    act_open_site: _site_url,
    act_open_new_tab: _site_url,
}

HELP_LINES = [
    "найди <запрос> | найди на ютубе <запрос> | открой <сайт>",
    "нажми <текст ссылки> | следующая ссылка | ввод",
    "назад | вперед | обнови | стоп",
    "новая вкладка | закрой вкладку | вкладка <номер>",
    "вниз | вверх | ниже | выше | в начало | в конец",
    "увеличь | уменьши | полный экран | найди на странице <текст>",
    "напечатай <текст> | диктовка ... стоп диктовка",
    "пауза | играй | громче | тише | скриншот",
    "музыка: следующий / предыдущий трек | перемотай вперед/назад [на 30 секунд]",
    "    трек сначала | останови / включи музыку | что играет | открой яндекс музыку",
    "изменить ди-пи-ай <400-2500> | сменить руку | выключи жесты",
]


def word_eq(a, b):
    """Слова совпадают с точностью до окончания или пары букв: маленькая модель путает
    "новая"/"новые", "диктовка"/"диктовку", "ввод"/"взвод"."""
    if a == b:
        return True
    if a[:max(3, len(a) - 2)] == b[:max(3, len(b) - 2)]:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def words_match(said, phrase):
    return len(said) == len(phrase) and all(word_eq(a, b) for a, b in zip(said, phrase))


def number_of(word):
    if word.isdigit():
        return int(word)
    return NUMBERS.get(word) or next((n for w, n in NUMBERS.items() if word_eq(word, w)), None)


def parse_number(words):
    """Русское число словами или цифрами: ['тысяча', 'двести'] -> 1200. Посторонние слова пропускаются."""
    total, cur, found = 0, 0, False
    for w in words:
        if w.isdigit():
            cur, found = cur + int(w), True
        elif w in THOUSANDS:
            total, cur, found = total + (cur or 1) * 1000, 0, True
        elif w in NUM_WORDS:
            cur, found = cur + NUM_WORDS[w], True
    return total + cur if found else None


def dpi_request(words):
    """'изменить ди-пи-ай [число]' в любом искажении -> (True, число или None), иначе None."""
    if len(words) >= 2 and any(word_eq(words[0], v) for v in DPI_VERBS):
        second = words[1]
        if second.startswith(("д", "dpi", "чувствител")):
            return True, parse_number(words[2:])
        number = parse_number(words[1:])               # "ди-пи-ай" расслышали как число: "измените пять"
        if number is not None:
            return True, number
    if words and words[0] in ("ди", "дпи", "дтп", "dpi"):
        number = parse_number(words[1:])
        if number is not None:
            return True, number
    return None


def music_request(words):
    """Музыкальная команда -> (действие, аргумент) или None.

    "перемотай вперёд [на 30 секунд | на минуту]", "отмотай назад", "вперёд на 20 секунд",
    "следующий трек", "трек сначала", "что играет"...  Одиночные "вперёд"/"назад" остаются
    браузерными - перемотка только с глаголом или с указанием времени.
    """
    # из похожих фраз берём самую близкую: "включи музыку" и "выключи музыку" отличаются одной буквой
    best = None
    for action, phrases in MUSIC_PHRASES.items():
        for p in phrases:
            pw = p.split()
            if words_match(words, pw):
                score = sum(difflib.SequenceMatcher(None, a, b).ratio() for a, b in zip(words, pw))
                if best is None or score > best[0]:
                    best = (score, action)
    if best:
        return best[1], None
    has_verb = bool(words) and any(word_eq(words[0], v) for v in SEEK_VERBS)
    minutes = any(w.startswith("минут") for w in words)
    half = "полминуты" in words
    amount = parse_number(words)
    timed = amount is not None or minutes or half
    if has_verb and any(w in ("начало", "начала") for w in words):
        return "restart", None
    back = any(word_eq(w, "назад") for w in words)
    fwd = any(word_eq(w, "вперед") for w in words)
    if not has_verb and not ((back or fwd) and timed and "на" in words):
        return None
    seconds = 30 if half else (amount or 1) * 60 if minutes else amount or SEEK_DEFAULT
    backward = back or (not fwd and has_verb and word_eq(words[0], "отмотай"))
    return "seek", -seconds if backward else seconds


def parse(text):
    """Фраза -> (описание, куда, действие, аргумент) или None, если это не команда."""
    tokens = normalize(text).split()
    core = [(i, w) for i, w in enumerate(tokens) if w not in FILLERS]   # без "и", "ну", "пожалуйста"
    words = [w for _, w in core]
    if not words:
        return None
    dpi = dpi_request(words)
    if dpi:
        return "изменить DPI", None, act_dpi, dpi[1]
    music = music_request(words)
    if music:
        return f"музыка: {music[0]}", None, act_music, music
    system = voice_windows.system_request(words)    # громкость/яркость числом, встроенные программы
    if system:
        return system
    # короткие команды: фраза должна совпасть целиком, иначе обычная речь начнёт листать страницы
    best = None
    for phrase, cmd in FIXED_INDEX.items():
        pw = phrase.split()
        if words_match(words, pw):
            score = sum(difflib.SequenceMatcher(None, a, b).ratio() for a, b in zip(words, pw))
            if best is None or score > best[0]:
                best = (score, cmd)
    if best:
        desc, target, act = best[1]
        return desc, target, act, None
    # "вкладка три", "вкладка номер 5", "перейди на вкладку вторую"
    if 2 <= len(words) <= 4 and any(word_eq(w, "вкладка") for w in words[:-1]):
        n = number_of(words[-1])
        if n and 1 <= n <= 8:
            return f"вкладка {n}", "browser", act_tab, n
    # команды с продолжением: приставка сравнивается нечётко, продолжение берётся как есть
    for prefixes, desc, target, act in PREFIXED:
        for p in prefixes:
            pw = p.split()
            if len(words) > len(pw) and words_match(words[:len(pw)], pw):
                rest = tokens[core[len(pw) - 1][0] + 1:]
                while rest and rest[0] in FILLERS:
                    rest = rest[1:]
                if rest:
                    arg = " ".join(rest)
                    return f"{desc}: {arg}", target, act, arg
    return None


def format_dictation(text, capital, first):
    """Сказанное -> печатаемый текст: знаки препинания словами, заглавные после точки, пробелы."""
    for phrase, sym in PUNCTUATION:
        text = re.sub(rf"\s*\b{phrase}\b", sym, text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"([.!?]\s+|\n)(\w)", lambda m: m.group(1) + m.group(2).upper(), text)
    if capital and text[:1].isalpha():
        text = text[0].upper() + text[1:]
    if not first and text[:1] not in (".", ",", "!", "?", ":", ";", "\n"):
        text = " " + text
    return text, text.rstrip(" ").endswith((".", "!", "?", "\n"))


# ======================= окна =======================

def active_title():
    try:
        w = pygetwindow.getActiveWindow()
        return w.title if w else ""
    except Exception:
        return ""


_user32, _kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
_user32.GetForegroundWindow.restype = ctypes.wintypes.HWND
_user32.GetWindowThreadProcessId.argtypes = [ctypes.wintypes.HWND, ctypes.POINTER(ctypes.wintypes.DWORD)]


def window_exe(hwnd):
    """Имя программы, которой принадлежит окно: 'zen.exe', 'chrome.exe'..."""
    pid = ctypes.wintypes.DWORD()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    handle = _kernel32.OpenProcess(0x1000, False, pid.value)      # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ""
    buf, size = ctypes.create_unicode_buffer(260), ctypes.wintypes.DWORD(260)
    _kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
    _kernel32.CloseHandle(handle)
    return buf.value.rsplit("\\", 1)[-1].lower()


def is_browser(hwnd, title=""):
    # по имени программы надёжнее: у многих браузеров (Zen, Arc...) в заголовке нет своего названия
    return window_exe(hwnd) in BROWSER_EXES or any(b in title for b in BROWSER_TITLES)


def active_is_browser():
    hwnd = _user32.GetForegroundWindow()
    return bool(hwnd) and is_browser(hwnd, active_title())


def bring_to_front(hwnd):
    """Выводит окно на передний план. Windows запрещает это фоновым программам, поэтому
    подключаемся к очереди ввода активного окна, а если не помогло - "нажимаем" Alt."""
    if _user32.IsIconic(hwnd):
        _user32.ShowWindow(hwnd, 9)                                  # SW_RESTORE
    fg = _user32.GetForegroundWindow()
    fg_thread = _user32.GetWindowThreadProcessId(fg, None) if fg else 0
    me = _kernel32.GetCurrentThreadId()
    attached = bool(fg_thread and fg_thread != me and _user32.AttachThreadInput(fg_thread, me, True))
    _user32.BringWindowToTop(hwnd)
    _user32.SetForegroundWindow(hwnd)
    if attached:
        _user32.AttachThreadInput(fg_thread, me, False)
    if _user32.GetForegroundWindow() != hwnd:
        _user32.keybd_event(0x12, 0, 0, 0)                           # Alt вниз-вверх снимает запрет
        _user32.keybd_event(0x12, 0, 2, 0)
        _user32.SetForegroundWindow(hwnd)
    time.sleep(0.15)
    return _user32.GetForegroundWindow() == hwnd


def focus_browser():
    """Делает активным окно браузера (верхнее из открытых). True, если получилось."""
    if active_is_browser():
        return True
    for w in pygetwindow.getAllWindows():      # идут сверху вниз - первое найденное и есть последнее активное
        if w.title and w.visible and w.width > 200 and is_browser(w._hWnd, w.title):
            return bring_to_front(w._hWnd)
    return False


def wait_for_browser(secs):
    end = time.time() + secs
    while time.time() < end:
        if focus_browser():
            return True
        time.sleep(0.4)
    return False


# ======================= исполнитель =======================

class VoiceExecutor(threading.Thread):
    """Берёт фразы у VoiceListener, понимает команды и выполняет их (в своём потоке - камера не тормозит)."""

    def __init__(self, listener, controller=None):
        super().__init__(daemon=True)
        self.listener = listener
        self.controller = controller      # Controller из hand_control - для "выключи/включи жесты"
        self.dictation = False
        self.capital, self.first_word = True, True
        self.feedback = ("", 0.0, True)    # (текст, до какого времени показывать, успех)
        self.help_until = 0.0
        self.await_dpi_until = 0.0         # после "изменить ди-пи-ай" ждём число до этого времени
        self.music = YandexMusic()
        self.pending_confirm = None        # (действие, до какого времени ждать "да") - для опасных команд
        self._voice = None                 # синтезатор речи Windows для ответов вслух
        self.running = True

    def run(self):
        while self.running:
            try:
                text = self.listener.texts.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.handle(text)
            except Exception as e:
                self.say(f"ошибка: {e}", False)

    def say(self, text, ok=True, secs=3):
        self.feedback = (text, time.time() + secs, ok)
        print(("  -> " if ok else "  ?? ") + text)

    def speak(self, text):
        """Ответ вслух голосом Windows (русский, если установлен). Свой голос микрофон не слушает."""
        listener = self.listener
        try:
            import comtypes
            import comtypes.client
            comtypes.CoInitialize()
            if self._voice is None:
                self._voice = comtypes.client.CreateObject("SAPI.SpVoice")
                voices = self._voice.GetVoices()
                for i in range(voices.Count):
                    if "419" in voices.Item(i).GetAttribute("Language"):    # 419 - код русского языка
                        self._voice.Voice = voices.Item(i)
                        break
            if hasattr(listener, "ignore_until"):
                listener.ignore_until = time.time() + 30
            self._voice.Speak(text)
        except Exception as e:
            print("  (не получилось сказать вслух:", e, ")")
        finally:
            if hasattr(listener, "ignore_until"):
                listener.ignore_until = time.time() + 0.8
            try:                                    # и всё, что микрофон успел принять за это время
                while True:
                    listener.texts.get_nowait()
            except (queue.Empty, AttributeError):
                pass

    def report(self, result, desc):
        """Результат действия: None - успех, строка - ошибка, ("ok", текст) - успех, ("say", текст) - и вслух."""
        if isinstance(result, tuple):
            self.say(result[1], True, DPI_WAIT if self.await_dpi_until else 3)
            if result[0] == "say":
                self.speak(result[1])
        else:
            self.say(result or desc, result is None)

    def handle(self, text):
        t = normalize(text)
        if not t:
            return
        print(f"Слышу: {t}")
        if self.dictation:
            return self.dictate(t)
        if self.pending_confirm:                    # ждём "да" для выключения/перезагрузки
            run, until = self.pending_confirm
            self.pending_confirm = None
            words = [w for w in t.split() if w not in FILLERS]
            if time.time() < until and any(words_match(words, y.split()) for y in voice_windows.YES):
                return self.report(run(), "")
            self.say("отменено", False)
            if words in (["нет"], ["отмена"], ["не", "надо"]) or not words:
                return
        cmd = parse(t)
        if time.time() < self.await_dpi_until and (cmd is None or cmd[2] is act_dpi):
            return self.take_dpi(t)
        self.await_dpi_until = 0.0             # сказали другую команду - число больше не ждём
        if cmd is None:
            return self.say(f"не команда: «{t}»", False)
        desc, target, act, arg = cmd
        if target == "browser" and not focus_browser():
            if act in DIRECT_URL:                  # поиск и сайты откроем прямой ссылкой - сработает всегда
                webbrowser.open(DIRECT_URL[act](arg))
                return self.say(desc)
            act_open_browser(None, self)           # браузера нет - откроем и подождём его окно
            if not wait_for_browser(6):
                return self.say("не нашёл окно браузера", False)
        elif target == "text" and any(w in active_title() for w in OWN_WINDOWS):
            focus_browser()                        # из окна камеры печатать некуда - уйдём в браузер
        self.report(act(arg, self), desc)

    def take_dpi(self, t):
        """Ответ на "назовите DPI": число от DPI_MIN до DPI_MAX."""
        number = parse_number(t.split())
        if number is None:
            self.await_dpi_until = time.time() + DPI_WAIT
            return self.say(f"Не расслышал число. Назовите DPI от {DPI_MIN} до {DPI_MAX}", False, DPI_WAIT)
        if not DPI_MIN <= number <= DPI_MAX:
            self.await_dpi_until = time.time() + DPI_WAIT
            return self.say(f"{number} - вне диапазона {DPI_MIN}-{DPI_MAX}, повторите", False, DPI_WAIT)
        self.await_dpi_until = 0.0
        if self.controller is None:
            return self.say("жесты не запущены", False)
        self.controller.set_dpi(number)
        self.say(f"DPI = {number}")

    def dictate(self, t):
        words = [w for w in t.split() if w not in FILLERS]
        said = lambda *phrases: any(words_match(words, p.split()) for p in phrases)  # noqa: E731
        if said(*STOP_DICTATION):
            self.dictation = False
            return self.say("диктовка выключена")
        if said("ввод", "отправить", "отправь"):
            pyautogui.press("enter")
            self.capital, self.first_word = True, True
            return self.say("Enter")
        if said("стереть", "сотри", "удали слово"):
            pyautogui.hotkey("ctrl", "backspace")
            return self.say("стереть слово")
        if any(w in active_title() for w in OWN_WINDOWS):
            focus_browser()
        typed, self.capital = format_dictation(t, self.capital, self.first_word)
        self.first_word = False
        paste(typed)
        self.say(f"напечатано: {typed.strip()}")


# ======================= экранная подсказка (для окна камеры) =======================

_fonts = {}
_overlay = {}


def _font(size):
    if size not in _fonts:
        from PIL import ImageFont
        for name in ("segoeui.ttf", "arial.ttf"):
            try:
                _fonts[size] = ImageFont.truetype(str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name),
                                                  size)
                break
            except OSError:
                continue
        else:
            _fonts[size] = ImageFont.load_default()
    return _fonts[size]


def draw_voice_overlay(frame, listener, executor):
    """Рисует на кадре (BGR, numpy) статус голоса, распознаваемую фразу и результат команды."""
    import numpy as np
    from PIL import Image, ImageDraw
    now = time.time()
    status = {"on": "слушаю", "loading": "загрузка...", "error": "ошибка микрофона",
              "no vosk": "нет vosk (pip install vosk)"}.get(listener.status, listener.status)
    lines = [(f"Голос: {'ДИКТОВКА (скажите «стоп диктовка»)' if executor.dictation else status}", 20,
              (255, 220, 0) if executor.dictation else (0, 230, 120) if listener.status == "on" else (255, 160, 0))]
    music = getattr(executor, "music", None)
    if music is not None:
        music.watch()
        st = music.state
        if st:
            lines.append((f"Я.Музыка ({st['where']}): {'играет' if st['playing'] else 'пауза'} - {st['title']}"
                          f"{' - ' + st['artist'] if st['artist'] else ''}  {fmt(st['pos'])}/{fmt(st['dur'])}",
                          17, (255, 210, 120)))
    if listener.partial:
        lines.append((f"Слышу: {listener.partial}", 20, (220, 220, 220)))
    text, until, ok = executor.feedback
    if now < until:
        lines.append((text, 22, (120, 255, 120) if ok else (255, 150, 90)))
    if now < executor.help_until:
        lines += [(line, 15, (255, 255, 255)) for line in HELP_LINES]
    # текст с обводкой рисуется медленно, поэтому перерисовываем его только когда он изменился,
    # а на каждый кадр лишь накладываем готовую картинку
    h, w = frame.shape[:2]
    key = (tuple(lines), w, h)
    if _overlay.get("key") != key:
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        y = 50
        for text, size, color in lines:
            draw.text((10, y), text, font=_font(size), fill=color, stroke_width=2, stroke_fill=(0, 0, 0))
            y += size + 8
        rgba = np.array(img)[:min(h, y + 10)]
        _overlay.update(key=key, bgr=rgba[:, :, 2::-1].astype(np.float32),
                        alpha=rgba[:, :, 3:4].astype(np.float32) / 255)
    bgr, alpha = _overlay["bgr"], _overlay["alpha"]
    top = frame[:len(bgr)].astype(np.float32)
    frame[:len(bgr)] = (top + (bgr - top) * alpha).astype(np.uint8)
    return frame


def main():
    print(__doc__)
    listener = VoiceListener()
    listener.start()
    executor = VoiceExecutor(listener)
    executor.start()
    while listener.status == "loading":
        time.sleep(0.2)
    if listener.status != "on":
        raise SystemExit(f"Голос не запустился: {listener.status}")
    print("Слушаю. Скажите «команды» - список, Ctrl+C - выход.")
    executor.music.watch()
    shown = None
    try:
        while True:
            time.sleep(0.5)
            st = executor.music.state       # сообщаем, когда Яндекс Музыка появилась, пропала или сменился трек
            now = st and (st["where"], st["title"], st["playing"])
            if now != shown:
                shown = now
                print(f"[Я.Музыка: {st['where']}, {'играет' if st['playing'] else 'пауза'}: {st['title']} - {st['artist']}]"
                      if st else "[Я.Музыка не открыта]")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


# ======================= команды самой Windows =======================
# Подключаются в конце: voice_windows пользуется помощниками этого файла (paste, words_match...).
import voice_windows  # noqa: E402

FIXED += voice_windows.SYSTEM_FIXED
FIXED_INDEX = {phrase: cmd[1:] for cmd in FIXED for phrase in cmd[0]}
PREFIXED[:0] = voice_windows.SYSTEM_PREFIXED      # "открой папку ...", "запусти ..." - раньше общего "открой ..."
HELP_LINES += voice_windows.SYSTEM_HELP
