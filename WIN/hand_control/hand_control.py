# Copyright (c) 2026 Hand Control X authors. All rights reserved.
# Proprietary and confidential — see LICENSE. Do not redistribute.
"""
Управление компьютером жестами руки через веб-камеру.

Курсор работает как обычная мышь: двигаете руку - курсор сдвигается на столько же,
сколько сдвинулась бы мышь с заданным DPI (рука на 1 дюйм = DPI пикселей).

Жесты (одна рука):
  Ладонь                        -> курсор движется вместе с рукой
  Кулак                         -> "поднять мышь": курсор стоит, руку можно переставить
  OK: большой+указательный      -> левая кнопка: коснулись и отпустили - клик,
                                   коснулись и повели руку - перетаскивание
  Большой+средний               -> правый клик
  Большой+безымянный            -> двойной клик
  Указательный+средний (V)      -> прокрутка-джойстик: чем выше/ниже руку от места,
                                   где показали V, тем быстрее листается
  При щипке остальные пальцы держите поднятыми (как знак OK) - так щипок не путается с кулаком.

Жесты двумя руками (вторая рука того же человека - другая сторона, похожий размер, рядом):
  Оба щипка OK, развести/свести руки   -> масштаб (Ctrl + колесо): браузер, карты, картинки, документы
  Оба кулака ("взять окно")            -> окно под курсором переносится вместе с руками
  Обе ладони резко влево/вправо        -> соседний рабочий стол Windows
  Пока в кадре обе руки, курсор одной рукой не двигается.

Калибровка: при первом запуске (и по K или голосом "калибровка") программа просит раскрыть ладонь,
сжать кулак и сделать щипок - и подбирает пороги жестов под вашу руку и камеру.

Своя рука: программа запоминает руку, которой начали управлять (левая/правая, место, размер),
и не переключается на руки других людей в кадре. "Сменить руку" (голосом) или L - выбрать заново.

Голос (по-русски, офлайн через Vosk) - браузер и настройки, см. voice_browser.py:
  "изменить ди-пи-ай" -> затем назовите число от 400 до 2500 (можно сразу: "изменить ди-пи-ай тысяча")
  "сменить руку", "выключи жесты" / "включи жесты", "найди ...", "открой ютуб", "команды" - список.
  Windows (voice_windows.py): "запусти телеграм", "сверни окно", "громкость 40", "яркость 70",
  "который час", "заблокируй компьютер", "выключи компьютер" (переспросит, нужно сказать "да").

Клавиши в окне камеры:  P - пауза,  L - выбрать руку заново,  K - калибровка,  Q / Esc - выход
"""
import json
import math
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks.python import BaseOptions, vision

from voice_browser import DPI_MAX, DPI_MIN, VoiceExecutor, VoiceListener, draw_voice_overlay

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
SETTINGS_PATH = Path(__file__).with_name("hand_settings.json")

# пороги в 2D для игр (darts_game, jedi_duel), которые берут жесты отсюда
PINCH_ON, PINCH_OFF = 0.28, 0.40

DPI_DEFAULT = 1000
PALM_DEPTH_SMOOTH = 0.1       # сглаживание масштаба "пикселей кадра на метр" (рука редко резко меняет дистанцию)
PINCH3D_ON, PINCH3D_OFF = 0.30, 0.45   # щипок по 3D-расстоянию кончиков, в долях длины ладони
STRAIGHT_MAX_BEND = 70        # палец прямой, если суммарный изгиб двух суставов меньше, градусов
DEBOUNCE_FRAMES = 2           # новое состояние щипка должно продержаться столько кадров
CLICK_REWIND = 0.12           # при щипке курсор возвращается туда, где был за столько секунд до касания
PINCH_APPROACH = 0.7          # ...или туда, где пальцы только начали сходиться (расстояние < этой доли ладони)
DRAG_START_M = 0.012          # при зажатой кнопке рука сдвинулась на 1.2 см -> перетаскивание
SCROLL_DEADZONE = 0.25        # мёртвая зона джойстика прокрутки, в долях размера ладони
SCROLL_RATE = 2000            # единиц колеса в секунду на одну "ладонь" отклонения (120 = 1 щелчок)
EURO_MIN_CUTOFF, EURO_BETA = 0.5, 0.1    # фильтр One Euro: меньше cutoff - меньше дрожь, больше beta - меньше задержка
# Залипание в покое. Шум MediaPipe пропорционален размеру руки в кадре (модель смотрит на вырезанную
# руку одного размера), поэтому пороги - в долях длины ладони: одинаково работают вблизи и вдали.
DEADBAND = 0.011              # рука в покое: пока центр ладони гуляет в пределах ~1 мм, курсор стоит
REST_SPEED, REST_TIME = 0.18, 0.15   # "покой" - медленнее 0.18 ладони/с (~1.6 см/с) дольше 0.15 с
WARMUP = 0.2                  # после появления руки курсор стоит, пока фильтр не сойдётся
RELOCK_AFTER = 2.0            # рука пропала дольше - можно захватить её заново
INCH = 0.0254

CALIB_PREP, CALIB_REC = 1.5, 2.0   # калибровка: столько секунд на подготовку и запись каждого шага
CALIB_MIN_FRAMES = 15         # меньше кадров за шаг - шаг повторяется

PARTNER_SCALE = 0.45          # вторая рука того же человека: размер ладони отличается не больше чем в e^0.45 раз
PARTNER_DIST = 7.0            # ...и она не дальше 7 длин ладони от своей
ZOOM_STEP = 0.08              # двумя щипками: расстояние между руками изменилось на 8% -> шаг масштаба
SWIPE_DIST, SWIPE_SPEED = 1.5, 2.5   # взмах двумя ладонями: дальше 1.5 ладони и быстрее 2.5 ладони/с
SWIPE_COOLDOWN = 1.0

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, RING_TIP = 0, 4, 8, 9, 12, 16
FINGERS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}  # (pip, tip)
FINGER_JOINTS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
PALM_POINTS = (0, 5, 9, 13, 17)
CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


def ensure_model():
    if not MODEL_PATH.exists():
        print("Скачиваю модель MediaPipe hand_landmarker...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def extended(lm, finger):
    """Палец разогнут (2D-проверка, используется в играх)."""
    pip, tip = FINGERS[finger]
    return dist(lm[WRIST], lm[tip]) > dist(lm[WRIST], lm[pip]) * 1.1


def d3(a, b):
    return math.dist(a, b)


def bend(w, finger):
    """Суммарный изгиб двух суставов пальца по 3D-координатам, градусы (0 - палец прямой)."""
    a, b, c, d = (w[i] for i in FINGER_JOINTS[finger])
    total = 0.0
    for p, q, r in ((a, b, c), (b, c, d)):
        u = [q[k] - p[k] for k in range(3)]
        v = [r[k] - q[k] for k in range(3)]
        nu, nv = math.hypot(*u), math.hypot(*v)
        if nu > 1e-9 and nv > 1e-9:
            cos = sum(u[k] * v[k] for k in range(3)) / (nu * nv)
            total += math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    return total


def straight(w, finger, max_bend=STRAIGHT_MAX_BEND):
    """Палец прямой по 3D-координатам: не зависит от наклона и поворота руки."""
    return bend(w, finger) < max_bend


def px_per_meter(lm, w, frame_size):
    """Сколько пикселей кадра в одном метре на расстоянии руки: из 3D-размера ладони."""
    fw, fh = frame_size
    img = [(p.x * fw, p.y * fh) for p in lm]
    return max(math.dist(img[0], img[9]) / (d3(w[0], w[9]) or 1e-6),
               math.dist(img[5], img[17]) / (d3(w[5], w[17]) or 1e-6))


def load_settings():
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_setting(key, value):
    settings = load_settings()
    settings[key] = value
    try:
        SETTINGS_PATH.write_text(json.dumps(settings, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass


class OneEuro:
    """Фильтр One Euro: при медленном движении сильно сглаживает дрожь, при быстром - почти без задержки."""

    def __init__(self):
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(dt, cutoff):
        tau = 1 / (2 * math.pi * cutoff)
        return 1 / (1 + tau / dt)

    def __call__(self, x, t):
        if self.t is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        self.dx += self._alpha(dt, 1.0) * ((x - self.x) / dt - self.dx)
        cutoff = EURO_MIN_CUTOFF + EURO_BETA * abs(self.dx)
        self.x += self._alpha(dt, cutoff) * (x - self.x)
        return self.x


def palm_center(lm, aspect=1.0):
    return (sum(lm[i].x for i in PALM_POINTS) / 5 * aspect, sum(lm[i].y for i in PALM_POINTS) / 5)


def palm_size(lm, aspect=1.0):
    return math.hypot((lm[0].x - lm[9].x) * aspect, lm[0].y - lm[9].y)


class HandLock:
    """Следит за "своей" рукой, чтобы курсор не перескакивал на руку другого человека.

    Все руки в кадре ведутся как отдельные треки с номерами (сопоставление по месту с учётом
    скорости, размеру ладони и стороне). Своя рука - трек, выбранный в начале (самая крупная рука,
    т.е. ближе всех к камере). Рука, которую видели одновременно со своей, считается чужой и
    управление не перехватывает, даже если своя на время пропала. Заменить свою может только
    рука, появившаяся после её исчезновения там же и того же размера; через RELOCK_AFTER секунд -
    любая рука той же стороны, через 3 * RELOCK_AFTER - любая.
    """
    FORGET = 1.0      # чужие треки, не виденные дольше этого, забываются

    def __init__(self):
        self.reset()

    def reset(self):
        self.tracks = {}          # номер -> {pos, vel, size, label, seen, born}
        self.next_id = 0
        self.mine = None
        self.visible = {}         # номер трека -> индекс руки в текущем кадре
        self.partner_id = None    # вторая рука того же человека (для жестов двумя руками)

    @property
    def label(self):
        return self.tracks[self.mine]["label"] if self.mine in self.tracks else None

    @staticmethod
    def _cost(tr, pos, size, label, t):
        dt = t - tr["seen"]
        k = min(dt, 0.3)
        pred = (tr["pos"][0] + tr["vel"][0] * k, tr["pos"][1] + tr["vel"][1] * k)
        if label != tr["label"] and dt > 0.1:
            return None       # непрерывно видимой руке прощаем мигание метки, вернувшейся - нет
        jump = math.dist(pos, pred) / tr["size"]
        scale = abs(math.log(max(size, 1e-6) / tr["size"]))
        if jump > min(1.5 + 4 * dt, 4.0) or scale > 0.6:        # слишком далеко или другой размер
            return None
        return jump + 2 * scale + (0.7 if label != tr["label"] else 0)

    def pick(self, hands, t, aspect):
        """hands: [(ориентиры, 'Left'/'Right')] -> индекс своей руки или None."""
        dets = [(palm_center(lm, aspect), palm_size(lm, aspect), label) for lm, label in hands]
        # 1) сопоставляем руки в кадре с известными треками: сначала самые уверенные пары
        pairs = []
        for tid, tr in self.tracks.items():
            for di, d in enumerate(dets):
                c = self._cost(tr, *d, t)
                if c is not None:
                    pairs.append((c, tid, di))
        owner, used = {}, set()
        for c, tid, di in sorted(pairs):
            if tid not in used and di not in owner:
                owner[di] = tid
                used.add(tid)
        for di, (pos, size, label) in enumerate(dets):
            tid = owner.get(di)
            if tid is None:                                       # новая рука в кадре
                tid = self.next_id
                self.next_id += 1
                owner[di] = tid
                self.tracks[tid] = {"pos": pos, "vel": (0.0, 0.0), "size": size, "label": label, "seen": t, "born": t}
                continue
            tr = self.tracks[tid]
            dt = max(t - tr["seen"], 1e-3)
            v = ((pos[0] - tr["pos"][0]) / dt, (pos[1] - tr["pos"][1]) / dt)
            tr["vel"] = (tr["vel"][0] * 0.5 + v[0] * 0.5, tr["vel"][1] * 0.5 + v[1] * 0.5)
            tr["pos"], tr["size"], tr["seen"] = pos, tr["size"] * 0.8 + size * 0.2, t
        self.tracks = {tid: tr for tid, tr in self.tracks.items() if tid == self.mine or t - tr["seen"] < self.FORGET}
        visible = self.visible = {tid: di for di, tid in owner.items()}

        # 2) своя рука
        if self.mine not in self.tracks:                          # начало работы или "сменить руку"
            if not visible:
                return None
            self.mine = max(visible, key=lambda tid: self.tracks[tid]["size"])
            return visible[self.mine]
        if self.mine in visible:
            return visible[self.mine]
        me = self.tracks[self.mine]
        lost = t - me["seen"]
        # заменить свою может только рука, появившаяся ПОСЛЕ её исчезновения (не та, что была рядом)
        fresh = [tid for tid in visible if self.tracks[tid]["born"] > me["seen"]]
        near = [(c, tid) for tid in fresh if self.tracks[tid]["label"] == me["label"]
                and (c := self._cost(me, self.tracks[tid]["pos"], self.tracks[tid]["size"], me["label"], t)) is not None]
        if near:
            new = min(near)[1]
        elif lost > 3 * RELOCK_AFTER and visible:
            new = max(visible, key=lambda tid: self.tracks[tid]["size"])
        elif lost > RELOCK_AFTER and (same := [tid for tid in fresh if self.tracks[tid]["label"] == me["label"]]):
            new = max(same, key=lambda tid: self.tracks[tid]["size"])
        else:
            return None
        del self.tracks[self.mine]
        self.mine = new
        return visible[new]

    def partner(self):
        """Вторая рука того же человека в текущем кадре -> индекс или None.

        Другая сторона (левая при правой), похожий размер ладони (то же расстояние до камеры)
        и рядом со своей. Рука другого человека обычно дальше от камеры - она меньше и не подходит.
        """
        me = self.tracks.get(self.mine)
        if me is None or self.mine not in self.visible:
            self.partner_id = None
            return None
        best = None
        for tid, di in self.visible.items():
            tr = self.tracks[tid]
            if tid == self.mine or tr["label"] == me["label"]:
                continue
            scale = abs(math.log(max(tr["size"], 1e-6) / me["size"]))
            gap = math.dist(tr["pos"], me["pos"]) / me["size"]
            if scale > PARTNER_SCALE or gap > PARTNER_DIST:
                continue
            cost = gap + 3 * scale - (2 if tid == self.partner_id else 0)   # прежнюю напарницу держим
            if best is None or cost < best[0]:
                best = (cost, tid, di)
        self.partner_id = best[1] if best else None
        return best[2] if best else None


class Controller:
    def __init__(self):
        self.dpi = int(load_settings().get("dpi", DPI_DEFAULT))
        self.paused = False
        self.lock = HandLock()
        self.cur_x, self.cur_y = pyautogui.position()
        self.last_set = (int(self.cur_x), int(self.cur_y))
        self.hist = deque(maxlen=40)             # (t, x, y, расстояние щипка) - для отмотки при клике
        self.px_per_m = None
        self.pinch_dist = None                   # для подсказки на экране
        self._now = 0.0
        self.calibration_requested = False       # голосом "калибровка" -> главный цикл запустит мастер
        self.apply_calibration(load_settings().get("calibration", {}))
        self._reset_state()

    def apply_calibration(self, cal):
        """Пороги жестов: из калибровки под конкретную руку, а без неё - значения по умолчанию."""
        self.pinch_on = cal.get("pinch_on", PINCH3D_ON)
        self.pinch_off = cal.get("pinch_off", PINCH3D_OFF)
        self.max_bend = cal.get("max_bend", STRAIGHT_MAX_BEND)
        self.deadband = cal.get("deadband", DEADBAND)
        self.rest_speed = cal.get("rest_speed", REST_SPEED)

    def save_calibration(self, cal):
        self.apply_calibration(cal)
        save_setting("calibration", cal)

    def _reset_state(self):
        self.fx, self.fy = OneEuro(), OneEuro()
        self.prev = self.out = None
        self.resting, self.slow_since, self.start_t = True, 0.0, 0.0
        self.pinch = self.prev_pinch = None
        self._cand, self._cand_count, self._cand_since = None, 0, 0.0
        self.left_down = self.frozen = False
        self.drag_acc = 0.0
        self.freeze_until = 0.0
        self.scroll_anchor, self.scroll_acc = None, 0.0

    def release(self):
        if self.left_down:
            pyautogui.mouseUp()
        self._reset_state()

    def relock(self):
        self.release()
        self.lock.reset()

    def set_dpi(self, dpi):
        self.dpi = int(max(DPI_MIN, min(DPI_MAX, dpi)))
        save_setting("dpi", self.dpi)

    # --- курсор ---
    def move(self, dx, dy, t):
        actual = pyautogui.position()
        if math.dist(actual, self.last_set) > 3:  # двигали обычной мышью - продолжаем оттуда
            self.cur_x, self.cur_y = actual
        self.cur_x = max(0.0, min(SCREEN_W - 1.0, self.cur_x + dx))
        self.cur_y = max(0.0, min(SCREEN_H - 1.0, self.cur_y + dy))
        target = (int(round(self.cur_x)), int(round(self.cur_y)))
        if target != self.last_set:
            pyautogui.moveTo(*target)
            self.last_set = target
        self.hist.append((t, self.cur_x, self.cur_y, self.pinch_dist))

    def rewind(self, t):
        """Вернуть курсор туда, где он был до начала щипка: пальцы, сходясь, слегка дёргают руку.

        Берём более ранний из двух моментов: за CLICK_REWIND до первого кадра касания
        (а не до подтверждения щипка, которое приходит на пару кадров позже) и момент,
        когда пальцы только начали сходиться. Но не дальше 0.4 с назад.
        """
        when = self._cand_since - CLICK_REWIND
        approach = next((h[0] for h in reversed(self.hist) if h[3] is not None and h[3] > PINCH_APPROACH), None)
        if approach is not None:
            when = min(when, approach)
        when = max(when, t - 0.4)
        past = next((h for h in reversed(self.hist) if h[0] <= when), self.hist[0] if self.hist else None)
        if past:
            self.cur_x, self.cur_y = past[1], past[2]
            self.last_set = (int(round(self.cur_x)), int(round(self.cur_y)))
            pyautogui.moveTo(*self.last_set)

    # --- жесты ---
    def detect_pinch(self, w, palm, up):
        """Какой палец касается большого ('index' / 'middle' / 'ring' / None), с защитой от мигания."""
        tips = {"index": INDEX_TIP, "middle": MIDDLE_TIP, "ring": RING_TIP}
        d = {f: d3(w[THUMB_TIP], w[tip]) / palm for f, tip in tips.items()}
        self.pinch_dist = min(d.values())
        if self.pinch and d[self.pinch] < self.pinch_off:
            raw = self.pinch                       # уже зажатый щипок держим до порога отпускания
        else:
            # начать щипок можно только с поднятыми остальными пальцами - так он не путается с кулаком
            allowed = {"index": up["middle"] or up["ring"] or up["pinky"],
                       "middle": up["index"] or up["ring"] or up["pinky"],
                       "ring": up["index"] and up["middle"] and up["pinky"]}
            ok = {f: v for f, v in d.items() if allowed[f]}
            best = min(ok, key=ok.get) if ok else None
            raw = best if best and ok[best] < self.pinch_on else None
        if raw == self._cand:
            self._cand_count += 1
        else:
            self._cand, self._cand_count, self._cand_since = raw, 1, self._now
        if self._cand_count >= DEBOUNCE_FRAMES:
            self.pinch = self._cand
        return self.pinch

    def update(self, lm, wlm, t, frame_size):
        """Ориентиры своей руки (кадр и 3D в метрах) -> действие мыши, возвращает название жеста."""
        if self.paused:
            return "PAUSED"
        fw, fh = frame_size
        self._now = t
        w = [(p.x, p.y, p.z) for p in wlm]
        palm3 = d3(w[0], w[9]) or 1e-6
        up = {f: straight(w, f, self.max_bend) for f in FINGER_JOINTS}
        pinch = self.detect_pinch(w, palm3, up)
        started = pinch is not None and pinch != self.prev_pinch
        self.prev_pinch = pinch

        img = [(p.x * fw, p.y * fh) for p in lm]
        ratio = px_per_meter(lm, w, frame_size)
        self.px_per_m = ratio if self.px_per_m is None else self.px_per_m + PALM_DEPTH_SMOOTH * (ratio - self.px_per_m)

        cx = sum(img[i][0] for i in PALM_POINTS) / 5         # центр ладони почти не смещается при щипке
        cy = sum(img[i][1] for i in PALM_POINTS) / 5
        fx, fy = self.fx(cx, t), self.fy(cy, t)
        fx, fy = self.stick(fx, fy, t, palm3 * self.px_per_m)
        ddx, ddy = (fx - self.prev[0], fy - self.prev[1]) if self.prev else (0.0, 0.0)
        if t - self.start_t < WARMUP:
            ddx = ddy = 0.0
        self.prev = (fx, fy)
        gain = self.dpi / INCH / self.px_per_m              # пикселей экрана на пиксель кадра
        dx, dy = ddx * gain, ddy * gain
        moved_m = math.hypot(ddx, ddy) / self.px_per_m

        if pinch == "ring":
            if started:
                self.rewind(t)
                pyautogui.doubleClick()
                self.freeze_until = t + 0.3
            return "DOUBLE CLICK"

        if pinch is None and not any(up.values()):
            if self.left_down:
                pyautogui.mouseUp()
                self.left_down = self.frozen = False
            self.scroll_anchor = None
            return "CLUTCH"                                 # кулак: "подняли мышь"

        if pinch is None and up["index"] and up["middle"] and not up["ring"] and not up["pinky"]:
            return self.scroll(fy, palm3 * self.px_per_m, t)
        self.scroll_anchor = None

        gesture = "MOVE"
        if pinch == "index" and not self.left_down:
            self.rewind(t)
            pyautogui.mouseDown()
            self.left_down, self.frozen, self.drag_acc = True, True, 0.0
        elif pinch != "index" and self.left_down:
            pyautogui.mouseUp()
            self.left_down = self.frozen = False
            self.freeze_until = t + 0.1
        if pinch == "middle" and started:
            self.rewind(t)
            pyautogui.rightClick()
            self.freeze_until = t + 0.3
        if self.left_down:
            # пока рука почти не двигается, курсор стоит - клик попадает точно в цель;
            # повели руку дальше порога - начинается перетаскивание
            if self.frozen:
                self.drag_acc += moved_m
                self.frozen = self.drag_acc < DRAG_START_M
            gesture = "CLICK" if self.frozen else "DRAG"
        elif pinch == "middle":
            gesture = "RIGHT CLICK"
        if self.frozen or t < self.freeze_until:
            dx = dy = 0.0
        self.move(dx, dy, t)
        return gesture

    def stick(self, x, y, t, palm_px):
        """Залипание в покое: неподвижная рука не дрожит курсором, а движение передаётся без потерь.

        В покое курсор привязан к точке и стоит, пока рука не отойдёт от неё дальше DEADBAND.
        После этого курсор точно следует за рукой (весь путь, включая первый миллиметр), пока рука
        снова не остановится.
        """
        speed = math.hypot(self.fx.dx or 0.0, self.fy.dx or 0.0) / max(palm_px, 1e-6)
        if self.out is None:
            self.start_t = t
        if t - self.start_t < WARMUP:
            # рука только появилась: фильтр ещё сходится - точку покоя ставим, когда он успокоится
            self.out, self.resting, self.slow_since = (x, y), True, t
        elif self.resting:
            if math.dist((x, y), self.out) > self.deadband * palm_px:
                self.resting, self.out = False, (x, y)
        else:
            self.out = (x, y)
            if speed > self.rest_speed:
                self.slow_since = t
            elif t - self.slow_since > REST_TIME:
                self.resting = True
        return self.out

    def scroll(self, y, palm_px, t):
        """Джойстик: скорость прокрутки зависит от того, насколько рука выше/ниже исходной точки."""
        if self.scroll_anchor is None:
            self.scroll_anchor, self.scroll_t, self.scroll_acc = y, t, 0.0
            return "SCROLL"
        dt, self.scroll_t = t - self.scroll_t, t
        off = (self.scroll_anchor - y) / max(palm_px, 1e-6)
        excess = max(0.0, abs(off) - SCROLL_DEADZONE)
        self.scroll_acc += math.copysign(excess, off) * SCROLL_RATE * dt
        amount = int(self.scroll_acc)
        if amount:
            pyautogui.scroll(amount)
            self.scroll_acc -= amount
        return "SCROLL " + ("UP" if off > SCROLL_DEADZONE else "DOWN" if off < -SCROLL_DEADZONE else "")


def quantile(xs, q):
    xs = sorted(xs)
    return xs[int(q * (len(xs) - 1))]


class Calibration:
    """Мастер калибровки: подбирает пороги жестов под руку и камеру конкретного человека.

    Три шага по CALIB_REC секунд: раскрытая неподвижная ладонь, кулак, щипок OK. По записанным
    кадрам считаются:
      * max_bend   - граница "палец прямой / согнут": посередине между ладонью и кулаком;
      * pinch_on/off - пороги щипка: между расстоянием кончиков при щипке и при раскрытой ладони;
      * deadband, rest_speed - мёртвая зона и порог "покоя": по дрожи неподвижной руки
        (шум камеры + тремор), чтобы курсор не дрожал, но и не залипал зря.
    Если шаги не отличаются друг от друга (например, щипок не сделали), прежние пороги остаются.
    """
    STEPS = [("open", "Раскройте ладонь и держите её неподвижно"),
             ("fist", "Сожмите кулак"),
             ("pinch", "Щипок: соедините большой и указательный (знак OK)")]

    def __init__(self):
        self.active = False
        self.message, self.message_until, self.ok = "", 0.0, True

    def start(self):
        self.active, self.step, self.step_t = True, 0, None
        self.data = {name: [] for name, _ in self.STEPS}
        self.fx, self.fy = OneEuro(), OneEuro()

    def cancel(self, t):
        self.active = False
        self.message, self.message_until, self.ok = "Калибровка отменена", t + 3, False

    def lost(self):
        """Рука пропала посреди записи шага - шаг начинается заново."""
        if self.active and self.step_t is not None:
            self.step_t = None
            self.data[self.STEPS[self.step][0]].clear()

    def feed(self, lm, wlm, t, frame_size):
        """Кадр своей руки. Возвращает словарь порогов, когда калибровка закончилась успешно."""
        if self.step_t is None:
            self.step_t = t
            self.fx, self.fy = OneEuro(), OneEuro()
        elapsed = t - self.step_t
        if elapsed < CALIB_PREP:
            return None
        w = [(p.x, p.y, p.z) for p in wlm]
        palm3 = d3(w[0], w[9]) or 1e-6
        palm_px = palm3 * px_per_meter(lm, w, frame_size)
        fw, fh = frame_size
        cx = sum(lm[i].x for i in PALM_POINTS) / 5 * fw
        cy = sum(lm[i].y for i in PALM_POINTS) / 5 * fh
        fx, fy = self.fx(cx, t), self.fy(cy, t)                   # тот же фильтр, что у курсора
        self.data[self.STEPS[self.step][0]].append({
            "bends": {f: bend(w, f) for f in FINGER_JOINTS},
            "pinch": d3(w[THUMB_TIP], w[INDEX_TIP]) / palm3,
            "pos": (fx / palm_px, fy / palm_px),
            "speed": math.hypot(self.fx.dx, self.fy.dx) / palm_px,
        })
        if elapsed < CALIB_PREP + CALIB_REC:
            return None
        if len(self.data[self.STEPS[self.step][0]]) < CALIB_MIN_FRAMES:   # мало кадров - повторяем шаг
            self.lost()
            return None
        self.step, self.step_t = self.step + 1, None
        if self.step < len(self.STEPS):
            return None
        self.active = False
        result, problem = self.compute(self.data)
        if problem:
            self.message, self.message_until, self.ok = f"Калибровка не удалась: {problem}", t + 6, False
            return None
        self.message, self.message_until, self.ok = "Калибровка сохранена", t + 3, True
        return result

    @staticmethod
    def compute(data):
        """Записанные шаги -> (пороги, None) или (None, что пошло не так)."""
        opened, fist, pinch = data["open"], data["fist"], data["pinch"]
        b_open = quantile([b for s in opened for b in s["bends"].values()], 0.9)    # самый согнутый "прямой"
        b_fist = quantile([b for s in fist for b in s["bends"].values()], 0.1)      # самый прямой "согнутый"
        if b_fist - b_open < 30:
            return None, "кулак и ладонь почти не отличаются - сожмите пальцы сильнее"
        max_bend = max(40.0, min(120.0, (b_open + b_fist) / 2))
        d_open = quantile([s["pinch"] for s in opened], 0.1)
        d_pinch = quantile([s["pinch"] for s in pinch], 0.9)
        if d_open - d_pinch < 0.2:
            return None, "щипок не распознан - плотнее соедините большой и указательный"
        others_up = sum(1 for s in pinch if any(s["bends"][f] < max_bend for f in ("middle", "ring", "pinky")))
        if others_up < 0.6 * len(pinch):
            return None, "при щипке держите средний, безымянный и мизинец поднятыми"
        gap = d_open - d_pinch
        center = (sum(s["pos"][0] for s in opened) / len(opened), sum(s["pos"][1] for s in opened) / len(opened))
        wobble = max(math.dist(s["pos"], center) for s in opened)                  # дрожь неподвижной руки
        return {
            "pinch_on": round(d_pinch + 0.35 * gap, 3),
            "pinch_off": round(d_pinch + 0.6 * gap, 3),
            "max_bend": round(max_bend, 1),
            "deadband": round(max(0.006, min(0.03, 1.3 * wobble)), 4),
            "rest_speed": round(max(0.08, min(0.5, 1.5 * quantile([s["speed"] for s in opened], 0.95))), 3),
        }, None

    def overlay(self, frame, t):
        lines = []
        if self.active:
            name, text = self.STEPS[self.step]
            lines.append((f"Калибровка, шаг {self.step + 1} из {len(self.STEPS)}", 22, (255, 230, 120)))
            lines.append((text, 26, (255, 255, 255)))
            if self.step_t is None:
                lines.append(("Покажите руку в камеру", 18, (255, 170, 90)))
            else:
                left = CALIB_PREP - (t - self.step_t)
                lines.append((f"Приготовьтесь... {left:.1f}" if left > 0 else "Запись, не двигайте рукой",
                              18, (200, 200, 200) if left > 0 else (120, 255, 120)))
            lines.append(("K - отменить", 15, (180, 180, 180)))
        elif t < self.message_until:
            lines.append((self.message, 22, (120, 255, 120) if self.ok else (255, 150, 90)))
        if lines:
            put_text(frame, lines)          # внизу, над строкой с клавишами


_fonts = {}


def _font(size):
    from PIL import ImageFont
    if size not in _fonts:
        try:
            _fonts[size] = ImageFont.truetype("segoeui.ttf", size)
        except OSError:
            _fonts[size] = ImageFont.load_default()
    return _fonts[size]


def put_text(frame, lines, x0=10, y0=None, bottom=35):
    """Русский текст на кадре (cv2.putText кириллицу не умеет): [(текст, размер, (r, g, b))].
    Длинные строки переносятся по ширине кадра. Без y0 блок ставится внизу, на bottom пикселей выше края."""
    import numpy as np
    from PIL import Image, ImageDraw
    width = frame.shape[1] - x0 - 10
    wrapped = []
    for text, size, color in lines:
        line = ""
        for word in text.split():
            test = f"{line} {word}".strip()
            if line and _font(size).getlength(test) > width:
                wrapped.append((line, size, color))
                line = word
            else:
                line = test
        wrapped.append((line, size, color))
    height = sum(size + 8 for _, size, _ in wrapped) + 8
    y0 = max(0, min(frame.shape[0] - bottom - height if y0 is None else y0, frame.shape[0] - height))
    region = frame[y0:y0 + height]
    img = Image.fromarray(np.ascontiguousarray(region[:, :, ::-1]))
    draw = ImageDraw.Draw(img)
    y = 4
    for text, size, color in wrapped:
        draw.text((x0, y), text, font=_font(size), fill=color, stroke_width=2, stroke_fill=(0, 0, 0))
        y += size + 8
    frame[y0:y0 + height] = np.array(img)[:, :, ::-1]


class TwoHands:
    """Жесты двумя руками: масштаб, перенос окна, переключение рабочих столов."""

    def __init__(self):
        self.engaged = False      # в прошлом кадре было две руки (одна рука в это время курсор не ведёт)
        self.reset()

    def reset(self):
        self.mode = None
        self.fx, self.fy, self.fd = OneEuro(), OneEuro(), OneEuro()
        self.hist = deque(maxlen=20)
        self.swipe_until = getattr(self, "swipe_until", 0.0)
        self.window = None

    @staticmethod
    def pose(lm, wlm, ctrl, frame_size, mode):
        w = [(p.x, p.y, p.z) for p in wlm]
        palm3 = d3(w[0], w[9]) or 1e-6
        up = {f: straight(w, f, ctrl.max_bend) for f in FINGER_JOINTS}
        d = d3(w[THUMB_TIP], w[INDEX_TIP]) / palm3
        fw, fh = frame_size
        info = {"center": (sum(lm[i].x for i in PALM_POINTS) / 5 * fw, sum(lm[i].y for i in PALM_POINTS) / 5 * fh),
                "px_per_m": px_per_meter(lm, w, frame_size), "palm3": palm3}
        if d < (ctrl.pinch_off if mode == "zoom" else ctrl.pinch_on):
            info["pose"] = "pinch"
        elif not any(up.values()):
            info["pose"] = "fist"
        elif all(up.values()):
            info["pose"] = "open"
        else:
            info["pose"] = "other"
        return info

    def update(self, mine, other, t, frame_size, ctrl):
        """mine, other: (ориентиры, 3D-ориентиры) своей руки и второй. Возвращает название жеста."""
        self.engaged = True
        a = self.pose(*mine, ctrl, frame_size, self.mode)
        b = self.pose(*other, ctrl, frame_size, self.mode)
        px_per_m = (a["px_per_m"] + b["px_per_m"]) / 2
        palm_px = (a["palm3"] + b["palm3"]) / 2 * px_per_m
        poses = (a["pose"], b["pose"])
        if poses == ("pinch", "pinch"):
            # расстояние между центрами ладоней (кончики при щипке смотрят друг на друга и почти сходятся),
            # не меньше двух ладоней - иначе руки рядом давали бы огромные относительные изменения
            return self.zoom(max(math.dist(a["center"], b["center"]), 2 * palm_px), t)
        mid = ((a["center"][0] + b["center"][0]) / 2, (a["center"][1] + b["center"][1]) / 2)
        if poses == ("fist", "fist"):
            return self.grab(mid, t, ctrl.dpi / INCH / px_per_m)
        if self.mode is not None:
            self.reset()
        if poses == ("open", "open"):
            return self.swipe(mid[0] / palm_px, t)
        self.hist.clear()
        return "TWO HANDS"

    def zoom(self, d, t):
        """Развели руки на ZOOM_STEP - шаг увеличения (Ctrl + колесо вверх), свели - уменьшения."""
        if self.mode != "zoom":
            self.reset()
            self.mode, self.ref = "zoom", self.fd(d, t)
            return "ZOOM"
        d = self.fd(d, t)
        steps = 0
        while d > self.ref * (1 + ZOOM_STEP) and steps < 3:
            self.ref *= 1 + ZOOM_STEP
            steps += 1
        while d < self.ref / (1 + ZOOM_STEP) and steps > -3:
            self.ref /= 1 + ZOOM_STEP
            steps -= 1
        if steps:
            pyautogui.keyDown("ctrl")
            pyautogui.scroll(120 * steps)
            pyautogui.keyUp("ctrl")
        return "ZOOM " + ("IN" if steps > 0 else "OUT" if steps < 0 else "")

    def grab(self, mid, t, gain):
        """Два кулака: окно под курсором едет вместе с серединой между руками (с тем же DPI, что курсор)."""
        if self.mode != "grab":
            self.reset()
            self.mode = "grab"
            self.anchor = (self.fx(mid[0], t), self.fy(mid[1], t))
            self.window = window_at(*pyautogui.position())
            if self.window is not None:
                if self.window.isMaximized:          # развёрнутое окно не двигается - сначала обычный размер
                    self.window.restore()
                self.win0 = (self.window.left, self.window.top)
            return "GRAB WINDOW" if self.window else "GRAB: no window under cursor"
        if self.window is None:
            return "GRAB: no window under cursor"
        mx, my = self.fx(mid[0], t), self.fy(mid[1], t)
        dx, dy = (mx - self.anchor[0]) * gain, (my - self.anchor[1]) * gain
        try:
            self.window.moveTo(int(self.win0[0] + dx), int(self.win0[1] + dy))
        except Exception:
            self.window = None
        return "MOVE WINDOW"

    def swipe(self, x, t):
        """Резкий взмах двумя раскрытыми ладонями: влево - следующий рабочий стол, вправо - предыдущий."""
        self.hist.append((t, x))
        while self.hist and t - self.hist[0][0] > 0.35:
            self.hist.popleft()
        if len(self.hist) >= 3 and t >= self.swipe_until:
            (t0, x0), (t1, x1) = self.hist[0], self.hist[-1]
            dx = x1 - x0
            if abs(dx) > SWIPE_DIST and abs(dx) / max(t1 - t0, 1e-3) > SWIPE_SPEED:
                pyautogui.hotkey("winleft", "ctrl", "right" if dx < 0 else "left")
                self.swipe_until, self.hist = t + SWIPE_COOLDOWN, deque(maxlen=20)
                return "SWIPE " + ("NEXT DESKTOP" if dx < 0 else "PREV DESKTOP")
        return "TWO HANDS"


def window_at(x, y):
    """Верхнее окно под точкой экрана, кроме окна камеры и рабочего стола."""
    import pygetwindow
    for w in pygetwindow.getWindowsAt(x, y):
        if w.title and w.visible and not w.isMinimized and "Hand Control" not in w.title \
                and w.title != "Program Manager":
            return w
    return None


class Camera(threading.Thread):
    """Читает камеру в отдельном потоке: обработка всегда берёт самый свежий кадр,
    а не ждёт следующий и не обрабатывает устаревший из очереди."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.frame, self.t, self.n = None, 0.0, 0
        self.cond = threading.Condition()
        self.running = True

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                break
            with self.cond:
                self.frame, self.t, self.n = frame, time.time(), self.n + 1
                self.cond.notify_all()
        with self.cond:
            self.running = False
            self.cond.notify_all()

    def latest(self, after):
        """Ждёт кадр новее номера after -> (номер, время съёмки, кадр) или None, если камера отвалилась."""
        with self.cond:
            self.cond.wait_for(lambda: self.n > after or not self.running, timeout=2)
            return (self.n, self.t, self.frame) if self.n > after else None


def draw(frame, hands, mine, gesture, ctrl, partner=None):
    h, w = frame.shape[:2]
    for i, (lm, label) in enumerate(hands):
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        own = i in (mine, partner)
        color = (0, 255, 0) if i == mine else (255, 200, 0)          # своя - зелёная, вторая своя - голубая
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color if own else (130, 130, 130), 2 if own else 1)
        if own:
            for p in pts:
                cv2.circle(frame, p, 4, (0, 0, 255), -1)
            cv2.putText(frame, f"MY {label.upper()}", (pts[0][0] - 40, pts[0][1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    color = (0, 0, 255) if ctrl.paused else (0, 255, 255)
    cv2.putText(frame, gesture, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    info = f"DPI {ctrl.dpi}"
    if ctrl.pinch_dist is not None and mine is not None:
        info += f"   pinch {ctrl.pinch_dist:.2f} (on<{ctrl.pinch_on:.2f})"
    others = len(hands) - (mine is not None) - (partner is not None)
    if others:
        info += f"   other hands: {others}"
    cv2.putText(frame, info, (w - 10 - 9 * len(info), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "P: pause   L: re-pick hand   K: calibrate   Q/Esc: quit", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def main():
    ensure_model()
    options = vision.HandLandmarkerOptions(
        # буфер вместо пути: MediaPipe не всегда открывает файлы по путям с кириллицей
        base_options=BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=4,             # видим все руки в кадре и сами выбираем свою, а не "первую попавшуюся"
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # MJPG даёт больше кадров/с у многих камер
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    if not cap.isOpened():
        raise SystemExit("Не удалось открыть веб-камеру")

    ctrl = Controller()
    voice = VoiceListener()
    voice.start()
    executor = VoiceExecutor(voice, ctrl)   # команды выполняются в своём потоке - камера не тормозит
    executor.start()
    camera = Camera(cap)
    camera.start()
    calib, two = Calibration(), TwoHands()
    if "calibration" not in load_settings():
        ctrl.calibration_requested = True      # первый запуск - сначала подстроимся под руку
    t0, last_ts, last_n = time.time(), -1, 0
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            got = camera.latest(last_n)
            if got is None:
                break
            last_n, t, frame = got
            frame = cv2.flip(frame, 1)  # зеркально, чтобы движение совпадало с экраном
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = max(int((t - t0) * 1000), last_ts + 1)
            last_ts = ts
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            hands = [(lm, hd[0].category_name) for lm, hd in zip(result.hand_landmarks, result.handedness)]
            world = result.hand_world_landmarks
            mine = ctrl.lock.pick(hands, t, w / h)
            partner = ctrl.lock.partner() if mine is not None else None
            if ctrl.calibration_requested:
                ctrl.calibration_requested = False
                ctrl.release()
                calib.start()
            if calib.active:
                if mine is not None:
                    cal = calib.feed(hands[mine][0], world[mine], t, (w, h))
                    if cal:
                        ctrl.save_calibration(cal)
                else:
                    calib.lost()
                gesture = "CALIBRATION"
            elif mine is not None and partner is not None and not ctrl.paused:
                if not two.engaged:
                    ctrl.release()             # вторая рука в кадре: курсор одной рукой больше не ведём
                gesture = two.update((hands[mine][0], world[mine]), (hands[partner][0], world[partner]),
                                     t, (w, h), ctrl)
            else:
                if two.engaged:
                    two.engaged = False
                    two.reset()
                if mine is not None:
                    gesture = ctrl.update(hands[mine][0], world[mine], t, (w, h))
                else:
                    ctrl.release()
                    gesture = "PAUSED" if ctrl.paused else ("waiting for my hand" if ctrl.lock.label else "no hand")

            draw(frame, hands, mine, gesture, ctrl, partner)
            calib.overlay(frame, t)
            frame = draw_voice_overlay(frame, voice, executor)
            cv2.imshow("Hand Control", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                ctrl.paused = not ctrl.paused
                ctrl.release()
            if key == ord("l"):
                ctrl.relock()
            if key == ord("k"):
                if calib.active:
                    calib.cancel(t)
                else:
                    ctrl.calibration_requested = True

    voice.running = executor.running = camera.running = False
    ctrl.release()
    camera.join(timeout=1)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
