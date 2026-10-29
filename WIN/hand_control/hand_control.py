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

Своя рука: программа запоминает руку, которой начали управлять (левая/правая, место, размер),
и не переключается на руки других людей в кадре. "Сменить руку" (голосом) или L - выбрать заново.

Голос (по-русски, офлайн через Vosk) - браузер и настройки, см. voice_browser.py:
  "изменить ди-пи-ай" -> затем назовите число от 400 до 2500 (можно сразу: "изменить ди-пи-ай тысяча")
  "сменить руку", "выключи жесты" / "включи жесты", "найди ...", "открой ютуб", "команды" - список.
  Windows (voice_windows.py): "запусти телеграм", "сверни окно", "громкость 40", "яркость 70",
  "который час", "заблокируй компьютер", "выключи компьютер" (переспросит, нужно сказать "да").

Клавиши в окне камеры:  P - пауза,  L - выбрать руку заново,  Q / Esc - выход
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


def straight(w, finger):
    """Палец прямой по 3D-координатам: не зависит от наклона и поворота руки."""
    a, b, c, d = (w[i] for i in FINGER_JOINTS[finger])
    bend = 0.0
    for p, q, r in ((a, b, c), (b, c, d)):
        u = [q[k] - p[k] for k in range(3)]
        v = [r[k] - q[k] for k in range(3)]
        nu, nv = math.hypot(*u), math.hypot(*v)
        if nu > 1e-9 and nv > 1e-9:
            cos = sum(u[k] * v[k] for k in range(3)) / (nu * nv)
            bend += math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    return bend < STRAIGHT_MAX_BEND


def load_settings():
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


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
        visible = {tid: di for di, tid in owner.items()}

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
        self._reset_state()

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
        settings = load_settings()
        settings["dpi"] = self.dpi
        try:
            SETTINGS_PATH.write_text(json.dumps(settings), encoding="utf-8")
        except OSError:
            pass

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
        if self.pinch and d[self.pinch] < PINCH3D_OFF:
            raw = self.pinch                       # уже зажатый щипок держим до PINCH3D_OFF
        else:
            # начать щипок можно только с поднятыми остальными пальцами - так он не путается с кулаком
            allowed = {"index": up["middle"] or up["ring"] or up["pinky"],
                       "middle": up["index"] or up["ring"] or up["pinky"],
                       "ring": up["index"] and up["middle"] and up["pinky"]}
            ok = {f: v for f, v in d.items() if allowed[f]}
            best = min(ok, key=ok.get) if ok else None
            raw = best if best and ok[best] < PINCH3D_ON else None
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
        up = {f: straight(w, f) for f in FINGER_JOINTS}
        pinch = self.detect_pinch(w, palm3, up)
        started = pinch is not None and pinch != self.prev_pinch
        self.prev_pinch = pinch

        # сколько пикселей кадра в одном метре на расстоянии руки: из 3D-размера ладони
        img = [(p.x * fw, p.y * fh) for p in lm]
        ratio = max(math.dist(img[0], img[9]) / palm3, math.dist(img[5], img[17]) / (d3(w[5], w[17]) or 1e-6))
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
            if math.dist((x, y), self.out) > DEADBAND * palm_px:
                self.resting, self.out = False, (x, y)
        else:
            self.out = (x, y)
            if speed > REST_SPEED:
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


def draw(frame, hands, mine, gesture, ctrl):
    h, w = frame.shape[:2]
    for i, (lm, label) in enumerate(hands):
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        own = i == mine
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0) if own else (130, 130, 130), 2 if own else 1)
        if own:
            for p in pts:
                cv2.circle(frame, p, 4, (0, 0, 255), -1)
            cv2.putText(frame, f"MY {label.upper()}", (pts[0][0] - 40, pts[0][1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    color = (0, 0, 255) if ctrl.paused else (0, 255, 255)
    cv2.putText(frame, gesture, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    info = f"DPI {ctrl.dpi}"
    if ctrl.pinch_dist is not None and mine is not None:
        info += f"   pinch {ctrl.pinch_dist:.2f} (on<{PINCH3D_ON})"
    if len(hands) > 1:
        info += f"   other hands: {len(hands) - (mine is not None)}"
    cv2.putText(frame, info, (w - 10 - 9 * len(info), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "P: pause   L: re-pick hand   Q/Esc: quit", (10, h - 15),
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
            mine = ctrl.lock.pick(hands, t, w / h)
            if mine is not None:
                gesture = ctrl.update(hands[mine][0], result.hand_world_landmarks[mine], t, (w, h))
            else:
                ctrl.release()
                gesture = "PAUSED" if ctrl.paused else ("waiting for my hand" if ctrl.lock.label else "no hand")

            draw(frame, hands, mine, gesture, ctrl)
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

    voice.running = executor.running = camera.running = False
    ctrl.release()
    camera.join(timeout=1)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
