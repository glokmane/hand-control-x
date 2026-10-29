# Copyright (c) 2026 Hand Control X authors. All rights reserved.
# Proprietary and confidential — see LICENSE. Do not redistribute.
"""
Мини 3D-дартс с управлением рукой через веб-камеру (основа - hand_control.py).

Как бросать:
  1. Медленно двигайте открытую ладонь - жёлтый прицел ходит по мишени.
  2. Щипок (большой + указательный) - взять дротик.
  3. Резко взмахните рукой вверх и разожмите пальцы - бросок.
     Сила взмаха влияет на высоту попадания: слабо - ниже, сильно - выше
     (шкала справа показывает силу, зелёная зона - идеал).

5 раундов по 3 дротика. Клавиши: M - камера/мышь, R - заново, Esc - выход.
В режиме мыши: зажать ЛКМ, резко дёрнуть мышь вверх и отпустить.
"""
import json
import math
import queue
import random
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import pygame
from mediapipe.tasks.python import BaseOptions, vision
from OpenGL.GL import *  # noqa: F403
from OpenGL.GLU import *  # noqa: F403

from hand_control import (CONNECTIONS, INDEX_TIP, MIDDLE_MCP, MODEL_PATH, PINCH_OFF,
                          PINCH_ON, THUMB_TIP, WRIST, dist, ensure_model)

W, H = 1280, 720
FOV = 34

# --- сцена (метры, реальные размеры дартса) ---
BOARD_Z, BOARD_Y = -2.37, 1.73          # центр мишени: 2.37 м от линии броска, 1.73 м от пола
WALL_Z = BOARD_Z - 0.05
BACK_Z = WALL_Z + 0.01                  # задняя стенка шкафчика
EYE = (0.0, 1.62, 0.25)
ROOM_W, ROOM_H, ROOM_FRONT = 1.0, 2.4, 0.8
LAMP_Y, LAMP_Z = 2.15, BOARD_Z + 0.8
CAB_W, CAB_H, CAB_D = 0.31, 0.36, 0.11  # полуширина, полувысота, глубина шкафчика

BOARD_R, SURROUND_R, SCORE_R = 0.225, 0.245, 0.170
R_BULL, R_OUTER_BULL = 0.00635, 0.0159
R_TRIPLE, R_DOUBLE = (0.099, 0.107), (0.162, 0.170)
SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# --- физика и управление ---
G = 9.81
VZ = 9.0                  # скорость дротика к мишени при идеальном взмахе, м/с
IDEAL_FLICK = 1.6         # идеальная скорость взмаха рукой (долей кадра в секунду)
MOUSE_IDEAL_FLICK = 3.0   # то же для мыши (долей окна в секунду)
MIN_FLICK = 0.3           # слабее этой доли от идеала - бросок не засчитывается
POWER_SENS = 0.35         # насколько ошибка силы взмаха влияет на высоту
SIDE_ERR = 0.3            # насколько боковой увод руки при взмахе уводит дротик
AIM_SCALE = 0.9           # метров по мишени на всю ширину кадра
AIM_LOCK = 0.4            # при скорости руки выше этой доли от идеала прицел замирает

ROUNDS, DARTS_PER_ROUND = 5, 3
FLIGHT_COLORS = [(0.9, 0.15, 0.15), (0.15, 0.45, 0.95), (0.95, 0.8, 0.1)]
BEST_PATH = Path(__file__).with_name("darts_best.json")
PREVIEW_W, PREVIEW_H = 240, 180


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def score_at(x, y):
    """Очки за попадание в точку (x, y) относительно центра мишени."""
    r = math.hypot(x, y)
    if r <= R_BULL:
        return 50, "BULL"
    if r <= R_OUTER_BULL:
        return 25, "25"
    if r > SCORE_R:
        return 0, "Мимо"
    n = SECTORS[round((90 - math.degrees(math.atan2(y, x))) / 18) % 20]
    if R_TRIPLE[0] <= r <= R_TRIPLE[1]:
        return 3 * n, f"T{n}"
    if R_DOUBLE[0] <= r <= R_DOUBLE[1]:
        return 2 * n, f"D{n}"
    return n, str(n)


# ======================= ввод: камера или мышь =======================

class HandTracker(threading.Thread):
    """Фоновый поток: камера -> MediaPipe -> выборки (t, x, y, щипок, рука видна)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = queue.SimpleQueue()
        self.preview = None
        self.error = None
        self.ready = False
        self.running = True

    def run(self):
        try:
            ensure_model()
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
                running_mode=vision.RunningMode.VIDEO, num_hands=1,
                min_hand_detection_confidence=0.6, min_tracking_confidence=0.5)
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not cap.isOpened():
                self.error = "веб-камера не найдена"
                return
            with vision.HandLandmarker.create_from_options(options) as landmarker:
                self._loop(cap, landmarker)
            cap.release()
        except Exception as e:  # покажем ошибку в игре и переключимся на мышь
            self.error = str(e)

    def _loop(self, cap, landmarker):
        pinch, last_ts, t0 = False, -1, time.time()
        while self.running:
            ok, frame = cap.read()
            if not ok:
                continue
            t = time.time()
            frame = cv2.flip(frame, 1)
            ts = max(int((t - t0) * 1000), last_ts + 1)
            last_ts = ts
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(image, ts)
            lm = result.hand_landmarks[0] if result.hand_landmarks else None
            if lm:
                size = dist(lm[WRIST], lm[MIDDLE_MCP]) or 1e-6
                pinch = dist(lm[THUMB_TIP], lm[INDEX_TIP]) / size < (PINCH_OFF if pinch else PINCH_ON)
                self.samples.put((t, lm[MIDDLE_MCP].x, lm[MIDDLE_MCP].y, pinch, True))
            else:
                pinch = False
                self.samples.put((t, 0.5, 0.5, False, False))
            self.preview = self._make_preview(frame, lm, pinch)
            self.ready = True

    @staticmethod
    def _make_preview(frame, lm, pinch):
        small = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
        if lm:
            pts = [(int(p.x * PREVIEW_W), int(p.y * PREVIEW_H)) for p in lm]
            for a, b in CONNECTIONS:
                cv2.line(small, pts[a], pts[b], (0, 255, 0), 1)
            cv2.circle(small, pts[MIDDLE_MCP], 5, (0, 0, 255) if pinch else (0, 255, 255), -1)
        small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return cv2.flip(small, 0).tobytes()  # OpenGL рисует снизу вверх


class HandInput:
    """История положения руки (или мыши): позиция, щипок, скорость взмаха."""

    def __init__(self):
        self.hist = deque(maxlen=120)
        self.x = self.y = 0.5
        self.pinch = self.visible = False

    def reset(self):
        self.hist.clear()
        self.pinch = False

    def feed(self, t, x, y, pinch, visible):
        if visible:
            self.hist.append((t, x, y))
            self.x, self.y = x, y
        else:
            self.hist.clear()
        self.pinch = pinch and visible
        self.visible = visible

    def velocity(self, now, window=0.1):
        """Средняя скорость за последние window секунд; vy > 0 - вверх."""
        pts = [p for p in self.hist if now - p[0] <= window]
        if len(pts) < 2 or pts[-1][0] - pts[0][0] < 1e-3:
            return 0.0, 0.0
        (t0, x0, y0), (t1, x1, y1) = pts[0], pts[-1]
        return (x1 - x0) / (t1 - t0), (y0 - y1) / (t1 - t0)

    def flick(self):
        """Пиковая скорость взмаха вверх за последние 0.35 с (окна по 60 мс)."""
        if not self.hist:
            return 0.0, 0.0
        end = self.hist[-1][0]
        pts = [p for p in self.hist if end - p[0] <= 0.35]
        best = (0.0, 0.0)
        for i, (t0, x0, y0) in enumerate(pts):
            for t1, x1, y1 in pts[i + 1:]:
                if t1 - t0 >= 0.06:
                    v = ((x1 - x0) / (t1 - t0), (y0 - y1) / (t1 - t0))
                    if v[1] > best[1]:
                        best = v
                    break
        return best


# ======================= игровая логика =======================

class Game:
    def __init__(self):
        try:
            self.best = json.loads(BEST_PATH.read_text())["best"]
        except Exception:
            self.best = 0
        self.reset()

    def reset(self):
        self.round, self.dart_idx = 1, 0
        self.round_scores = [[] for _ in range(ROUNDS)]
        self.stuck = []            # воткнутые дротики: (цвет, позиция кончика, направление)
        self.flying = None
        self.state, self.state_time = "ready", time.time()
        self.message = ("Раунд 1", (1, 1, 1), time.time() + 1.5)
        self.aim = [0.0, BOARD_Y]
        self.hand_s = [0.5, 0.5]
        self.was_pinch = False
        self.last_power = None
        self.new_record = False

    @property
    def total(self):
        return sum(sum(r) for r in self.round_scores)

    def say(self, text, color=(1, 1, 1), secs=1.5):
        self.message = (text, color, time.time() + secs)

    def set_state(self, state):
        self.state, self.state_time = state, time.time()

    def eye(self):
        # камера чуть следует за прицелом - даёт параллакс и ощущение объёма
        return (EYE[0] + self.aim[0] * 0.35, EYE[1] + (self.aim[1] - BOARD_Y) * 0.3, EYE[2])

    def held_tip(self, pinch):
        ex, ey, ez = self.eye()
        hx, hy = self.hand_s
        return [ex + (hx - 0.5) * 0.24, ey - 0.07 - (hy - 0.5) * 0.10 + (0.02 if pinch else 0), ez - 0.5]

    def update(self, dt, inp, ideal):
        now = time.time()
        if inp.visible:
            k = min(1.0, dt * 20)
            self.hand_s[0] += (inp.x - self.hand_s[0]) * k
            self.hand_s[1] += (inp.y - self.hand_s[1]) * k
            # прицел двигается только при медленном движении руки - взмах его не сбивает
            if math.hypot(*inp.velocity(now)) < ideal * AIM_LOCK:
                tx = clamp((inp.x - 0.5) * AIM_SCALE, -0.4, 0.4)
                ty = BOARD_Y + clamp(-(inp.y - 0.5) * AIM_SCALE, -0.4, 0.4)
                k = min(1.0, dt * 10)
                self.aim[0] += (tx - self.aim[0]) * k
                self.aim[1] += (ty - self.aim[1]) * k

        pressed = inp.pinch and not self.was_pinch
        released = self.was_pinch and not inp.pinch
        self.was_pinch = inp.pinch
        elapsed = now - self.state_time

        if self.state == "ready" and released:
            if not inp.visible:
                self.say("Рука потеряна - бросок отменён", (1, 0.6, 0.3))
            else:
                vx, vy = inp.flick()
                if vy < ideal * MIN_FLICK:
                    self.say("Слабый взмах - резче!", (1, 0.6, 0.3), 2)
                else:
                    self.throw(vx / ideal, vy / ideal)
        elif self.state == "flying":
            self.step_flight(dt)
        elif self.state == "result" and elapsed > 0.9:
            if self.dart_idx < DARTS_PER_ROUND:
                self.set_state("ready")
            elif self.round < ROUNDS:
                self.say(f"Раунд {self.round}: {sum(self.round_scores[self.round - 1])} очков",
                         (0.6, 1, 0.6), 1.6)
                self.set_state("round_end")
            else:
                self.finish()
        elif self.state == "round_end" and elapsed > 1.6:
            self.round += 1
            self.dart_idx = 0
            self.stuck.clear()
            self.say(f"Раунд {self.round}")
            self.set_state("ready")
        elif self.state == "game_over" and pressed and elapsed > 1.5:
            self.reset()

    def throw(self, side, power):
        self.last_power = power
        p = 1 + (clamp(power, 0.5, 1.7) - 1) * POWER_SENS
        start = self.held_tip(True)
        t = (start[2] - BOARD_Z) / VZ
        # скорость, при которой идеальный взмах попадает точно в прицел (с учётом гравитации);
        # её масштабирование силой взмаха меняет время полёта, а значит и высоту попадания
        vel = [((self.aim[0] - start[0]) / t + side * SIDE_ERR) * p,
               ((self.aim[1] - start[1]) / t + 0.5 * G * t) * p,
               -VZ * p]
        self.flying = {"pos": start, "vel": vel, "spin": 0.0, "color": self.dart_idx}
        self.set_state("flying")

    def step_flight(self, dt):
        f = self.flying
        n = max(1, int(dt * 480))
        h = dt / n
        f["spin"] += dt * 720
        for _ in range(n):
            p, v = f["pos"], f["vel"]
            v[1] -= G * h
            q = [p[i] + v[i] * h for i in range(3)]
            if p[2] > BOARD_Z >= q[2]:
                hit = self.cross(p, q, BOARD_Z)
                r = math.hypot(hit[0], hit[1] - BOARD_Y)
                if r <= BOARD_R:
                    return self.land(hit, score_at(hit[0], hit[1] - BOARD_Y))
                if r <= SURROUND_R:
                    return self.land(hit, (0, "Мимо"))
            if q[2] <= BACK_Z:
                return self.land(self.cross(p, q, BACK_Z), (0, "Мимо"))
            if q[1] <= 0:
                return self.land(q, (0, "Недолёт"))
            f["pos"] = q

    @staticmethod
    def cross(p, q, z):
        k = (p[2] - z) / (p[2] - q[2])
        return [p[i] + (q[i] - p[i]) * k for i in range(3)]

    def land(self, hit, score):
        v = self.flying["vel"]
        n = math.sqrt(sum(c * c for c in v))
        d = [c / n for c in v]
        tip = [hit[i] + d[i] * 0.015 for i in range(3)]  # кончик входит в мишень на 1.5 см
        self.stuck.append((self.flying["color"], tip, d))
        self.flying = None
        pts, label = score
        self.round_scores[self.round - 1].append(pts)
        self.dart_idx += 1
        color = (1, 0.85, 0.2) if label[0] in "TDB" or label == "25" else (
            (1, 1, 1) if pts else (0.7, 0.7, 0.7))
        self.say(label if not pts or label == str(pts) else f"{label}   +{pts}", color, 1.2)
        self.set_state("result")

    def finish(self):
        self.new_record = self.total > self.best
        if self.new_record:
            self.best = self.total
            try:
                BEST_PATH.write_text(json.dumps({"best": self.best}))
            except OSError:
                pass
        self.set_state("game_over")


# ======================= графика =======================

def make_board_surface(size=1024):
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    c = size / 2
    k = c / BOARD_R
    black, cream, red, green = (25, 25, 25), (235, 220, 180), (200, 30, 35), (20, 130, 60)
    wire = (170, 170, 170)

    def pt(r, deg):
        return (c + r * k * math.cos(math.radians(deg)), c - r * k * math.sin(math.radians(deg)))

    pygame.draw.circle(surf, (18, 18, 18), (c, c), c)
    bands = [(R_OUTER_BULL, R_TRIPLE[0], False), (R_TRIPLE[0], R_TRIPLE[1], True),
             (R_TRIPLE[1], R_DOUBLE[0], False), (R_DOUBLE[0], R_DOUBLE[1], True)]
    font = pygame.font.SysFont("arial", 58, bold=True)
    for i, n in enumerate(SECTORS):
        mid = 90 - i * 18
        arc = [mid - 9 + 18 * j / 10 for j in range(11)]
        for r0, r1, ring in bands:
            color = (red if i % 2 == 0 else green) if ring else (black if i % 2 == 0 else cream)
            pygame.draw.polygon(surf, color, [pt(r1, a) for a in arc] + [pt(r0, a) for a in reversed(arc)])
        pygame.draw.line(surf, wire, pt(R_OUTER_BULL, mid - 9), pt(SCORE_R, mid - 9), 2)
        txt = pygame.transform.rotozoom(font.render(str(n), True, (240, 240, 240)), -i * 18, 1)
        surf.blit(txt, txt.get_rect(center=pt(0.197, mid)))
    for r in (*R_TRIPLE, *R_DOUBLE):
        pygame.draw.circle(surf, wire, (c, c), r * k, 2)
    pygame.draw.circle(surf, green, (c, c), R_OUTER_BULL * k)
    pygame.draw.circle(surf, red, (c, c), R_BULL * k)
    pygame.draw.circle(surf, wire, (c, c), R_OUTER_BULL * k, 2)
    return surf


def make_brick_surface(size=512):
    rnd = random.Random(7)
    surf = pygame.Surface((size, size))
    surf.fill((75, 68, 62))
    rows, bw = 8, size // 4
    bh = size // rows
    for r in range(rows):
        off = (bw // 2) * (r % 2)
        for col in range(-1, 5):
            s = rnd.uniform(0.7, 1.1)
            color = [min(255, int(v * s)) for v in (140, 62, 45)]
            pygame.draw.rect(surf, color, (col * bw + off + 3, r * bh + 3, bw - 6, bh - 6))
    return surf


def make_texture(surf, repeat=False):
    data = pygame.image.tobytes(surf, "RGBA", True)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    wrap = GL_REPEAT if repeat else GL_CLAMP_TO_EDGE
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, surf.get_width(), surf.get_height(),
                      GL_RGBA, GL_UNSIGNED_BYTE, data)
    return tex


def grid(origin, du, dv, nu, nv, normal, tile=None):
    """Плоскость, разбитая на сетку, - чтобы освещение от лампы было плавным пятном."""
    lu, lv = math.hypot(*du), math.hypot(*dv)
    glNormal3f(*normal)
    glBegin(GL_QUADS)
    for i in range(nu):
        for j in range(nv):
            for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                if tile:
                    glTexCoord2f(a / nu * lu / tile, b / nv * lv / tile)
                glVertex3f(*(origin[k] + du[k] * a / nu + dv[k] * b / nv for k in range(3)))
    glEnd()


def box(x0, y0, z0, x1, y1, z1):
    faces = [((0, 0, 1), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]),
             ((0, 0, -1), [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)]),
             ((1, 0, 0), [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]),
             ((-1, 0, 0), [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)]),
             ((0, 1, 0), [(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)]),
             ((0, -1, 0), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)])]
    glBegin(GL_QUADS)
    for n, verts in faces:
        glNormal3f(*n)
        for v in verts:
            glVertex3f(*v)
    glEnd()


class Scene:
    def __init__(self):
        self.q = gluNewQuadric()
        gluQuadricNormals(self.q, GLU_SMOOTH)
        self.board_tex = make_texture(make_board_surface())
        self.brick_tex = make_texture(make_brick_surface(), repeat=True)
        self.room = self.compile(self.build_room)
        self.board = self.compile(self.build_board)
        self.lamp = self.compile(self.build_lamp)
        self.darts = [self.compile(lambda c=c: self.build_dart(c)) for c in FLIGHT_COLORS]

    @staticmethod
    def compile(build):
        lst = glGenLists(1)
        glNewList(lst, GL_COMPILE)
        build()
        glEndList()
        return lst

    def build_room(self):
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.brick_tex)
        glColor3f(1, 1, 1)
        grid((-ROOM_W, 0, WALL_Z), (2 * ROOM_W, 0, 0), (0, ROOM_H, 0), 24, 24, (0, 0, 1), 0.6)
        grid((-ROOM_W, 0, ROOM_FRONT), (0, 0, WALL_Z - ROOM_FRONT), (0, ROOM_H, 0), 30, 24, (1, 0, 0), 0.6)
        grid((ROOM_W, 0, WALL_Z), (0, 0, ROOM_FRONT - WALL_Z), (0, ROOM_H, 0), 30, 24, (-1, 0, 0), 0.6)
        glDisable(GL_TEXTURE_2D)
        glColor3f(0.35, 0.22, 0.13)
        grid((-ROOM_W, 0, ROOM_FRONT), (2 * ROOM_W, 0, 0), (0, 0, WALL_Z - ROOM_FRONT), 20, 30, (0, 1, 0))
        glColor3f(0.18, 0.17, 0.16)
        grid((-ROOM_W, ROOM_H, WALL_Z), (2 * ROOM_W, 0, 0), (0, 0, ROOM_FRONT - WALL_Z), 20, 30, (0, -1, 0))

    def build_board(self):
        y0, y1, zf = BOARD_Y - CAB_H, BOARD_Y + CAB_H, WALL_Z + CAB_D
        t = 0.02
        glColor3f(0.32, 0.18, 0.09)                      # деревянный шкафчик
        box(-CAB_W, y0, WALL_Z, CAB_W, y1, BACK_Z)
        box(-CAB_W, y0, WALL_Z, -CAB_W + t, y1, zf)
        box(CAB_W - t, y0, WALL_Z, CAB_W, y1, zf)
        box(-CAB_W, y1 - t, WALL_Z, CAB_W, y1, zf)
        box(-CAB_W, y0, WALL_Z, CAB_W, y0 + t, zf)
        for side, angle in ((-1, 205), (1, 155)):        # открытые дверцы
            glPushMatrix()
            glTranslatef(side * CAB_W, 0, zf)
            glRotatef(angle, 0, 1, 0)
            x0, x1 = sorted((0, -side * CAB_W))
            glColor3f(0.32, 0.18, 0.09)
            box(x0, y0, -0.015, x1, y1, 0)
            glColor3f(0.1, 0.13, 0.12)                   # меловая доска на внутренней стороне
            box(x0 + 0.03, y0 + 0.04, -0.018, x1 - 0.03, y1 - 0.04, -0.014)
            glPopMatrix()

        glPushMatrix()                                   # чёрный обод мишени
        glTranslatef(0, BOARD_Y, BACK_Z)
        glColor3f(0.06, 0.06, 0.06)
        gluCylinder(self.q, SURROUND_R, SURROUND_R, BOARD_Z - BACK_Z, 64, 1)
        glTranslatef(0, 0, BOARD_Z - BACK_Z)
        gluDisk(self.q, 0, SURROUND_R, 64, 1)
        glPopMatrix()

        glEnable(GL_TEXTURE_2D)                          # сама мишень
        glBindTexture(GL_TEXTURE_2D, self.board_tex)
        glColor3f(1, 1, 1)
        glNormal3f(0, 0, 1)
        glBegin(GL_TRIANGLE_FAN)
        glTexCoord2f(0.5, 0.5)
        glVertex3f(0, BOARD_Y, BOARD_Z + 0.0015)
        for i in range(129):
            a = 2 * math.pi * i / 128
            glTexCoord2f(0.5 + 0.5 * math.cos(a), 0.5 + 0.5 * math.sin(a))
            glVertex3f(BOARD_R * math.cos(a), BOARD_Y + BOARD_R * math.sin(a), BOARD_Z + 0.0015)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def build_lamp(self):
        glPushMatrix()
        glTranslatef(0, LAMP_Y, LAMP_Z)
        glDisable(GL_LIGHTING)
        glColor3f(0.05, 0.05, 0.05)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, ROOM_H - LAMP_Y, 0)
        glEnd()
        glEnable(GL_LIGHTING)
        glRotatef(90, 1, 0, 0)                           # ось +z смотрит вниз
        glColor3f(0.1, 0.3, 0.2)
        gluCylinder(self.q, 0.035, 0.17, 0.14, 32, 1)
        glDisable(GL_LIGHTING)
        glColor3f(1, 0.95, 0.75)
        glTranslatef(0, 0, 0.09)
        gluSphere(self.q, 0.035, 16, 12)
        glEnable(GL_LIGHTING)
        glPopMatrix()

    def build_dart(self, flight):
        """Дротик: кончик в начале координат, смотрит в -z, хвост тянется в +z."""
        q = self.q
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.8, 0.8, 0.8, 1))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50)
        glPushMatrix()
        glColor3f(0.8, 0.8, 0.85)
        gluCylinder(q, 0.0003, 0.0013, 0.028, 12, 1)    # игла
        glTranslatef(0, 0, 0.028)
        glColor3f(0.42, 0.44, 0.48)                      # вольфрамовый корпус
        gluCylinder(q, 0.0022, 0.0036, 0.01, 16, 1)
        glTranslatef(0, 0, 0.01)
        gluCylinder(q, 0.0036, 0.0036, 0.03, 16, 1)
        glTranslatef(0, 0, 0.03)
        gluCylinder(q, 0.0036, 0.0017, 0.008, 16, 1)
        glTranslatef(0, 0, 0.008)
        glColor3f(0.12, 0.12, 0.12)                      # хвостовик
        gluCylinder(q, 0.0017, 0.0017, 0.04, 10, 1)
        glPopMatrix()
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0, 0, 0, 1))
        z0 = 0.096                                       # оперение - две скрещенные пластины
        glColor3f(*flight)
        glBegin(GL_QUADS)
        for ax, ay in ((1, 0), (0, 1)):
            glNormal3f(-ay, ax, 0)
            for s in (1, -1):
                for r, z in ((0.0017, z0), (0.016, z0 + 0.025), (0.016, z0 + 0.045), (0.0017, z0 + 0.045)):
                    glVertex3f(ax * s * r, ay * s * r, z)
        glEnd()

    def draw_dart(self, color, pos, d, spin=0.0, scale=1.0):
        yaw = math.degrees(math.atan2(-d[0], -d[2]))
        pitch = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))
        glPushMatrix()
        glTranslatef(*pos)
        glRotatef(yaw, 0, 1, 0)
        glRotatef(pitch, 1, 0, 0)
        glRotatef(spin, 0, 0, 1)
        glScalef(scale, scale, scale)
        glCallList(self.darts[color])
        glPopMatrix()

    def render(self, game, inp):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(FOV, W / H, 0.1, 20)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        eye = game.eye()
        gluLookAt(*eye, 0, BOARD_Y, BOARD_Z, 0, 1, 0)
        glLightfv(GL_LIGHT0, GL_POSITION, (0, LAMP_Y - 0.12, LAMP_Z, 1))

        glCallList(self.room)
        glCallList(self.board)
        glCallList(self.lamp)

        if game.state == "ready":                        # прицел на мишени
            glDisable(GL_LIGHTING)
            glColor4f(1, 0.9, 0.2, 0.9)
            ax, ay, z = game.aim[0], game.aim[1], BOARD_Z + 0.004
            glLineWidth(2)
            glBegin(GL_LINE_LOOP)
            for i in range(24):
                a = 2 * math.pi * i / 24
                glVertex3f(ax + 0.012 * math.cos(a), ay + 0.012 * math.sin(a), z)
            glEnd()
            glPointSize(4)
            glBegin(GL_POINTS)
            glVertex3f(ax, ay, z)
            glEnd()
            glEnable(GL_LIGHTING)

        # дротики вдали увеличены, иначе с 2.6 м их почти не видно
        for color, pos, d in game.stuck:
            self.draw_dart(color, pos, d, scale=1.6)
        if game.flying:
            f = game.flying
            self.draw_dart(f["color"], f["pos"], f["vel"], f["spin"], scale=1.6)
        if game.state == "ready" and inp.visible:        # дротик в руке, смотрит на прицел
            tip = game.held_tip(inp.pinch)
            d = [game.aim[0] - tip[0], game.aim[1] - tip[1], BOARD_Z - tip[2]]
            self.draw_dart(game.dart_idx, tip, d)


class Hud:
    def __init__(self):
        self.fonts = {}
        self.cache = {}

    def text(self, s, x, y, size=26, color=(1, 1, 1), center=False):
        """Текст в экранных координатах (y сверху), с тенью."""
        key = (s, size, color)
        if key not in self.cache:
            if len(self.cache) > 300:
                self.cache.clear()
            if size not in self.fonts:
                self.fonts[size] = pygame.font.SysFont("segoeui,arial", size, bold=True)
            rgb = tuple(int(c * 255) for c in color)
            fg = self.fonts[size].render(s, True, rgb)
            surf = pygame.Surface((fg.get_width() + 2, fg.get_height() + 2), pygame.SRCALPHA)
            surf.blit(self.fonts[size].render(s, True, (0, 0, 0)), (2, 2))
            surf.blit(fg, (0, 0))
            self.cache[key] = (pygame.image.tobytes(surf, "RGBA", True), surf.get_width(), surf.get_height())
        data, w, h = self.cache[key]
        if center:
            x -= w / 2
        glWindowPos2d(max(0, x), max(0, H - y - h))
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, data)

    @staticmethod
    def rect(x, y, w, h, color):
        glColor4f(*color)
        glBegin(GL_QUADS)
        glVertex2f(x, H - y)
        glVertex2f(x + w, H - y)
        glVertex2f(x + w, H - y - h)
        glVertex2f(x, H - y - h)
        glEnd()

    def draw(self, game, inp, mode, tracker, ideal):
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, W, 0, H, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        now = time.time()

        self.rect(12, 12, 300, 150, (0, 0, 0, 0.45))
        self.text(f"Раунд {min(game.round, ROUNDS)} / {ROUNDS}", 24, 18)
        self.text(f"Дротик {min(game.dart_idx + 1, DARTS_PER_ROUND)} / {DARTS_PER_ROUND}", 24, 48)
        self.text(f"Очки: {game.total}", 24, 78, color=(1, 0.85, 0.2))
        self.text(f"Рекорд: {game.best}", 24, 108, 22, (0.75, 0.75, 0.75))
        rounds = "  ".join(str(sum(r)) if r else "-" for r in game.round_scores)
        self.text(rounds, 24, 136, 18, (0.75, 0.75, 0.75))

        text, color, until = game.message
        if now < until and game.state != "game_over":
            self.text(text, W / 2, 40, 44, color, center=True)

        # шкала силы взмаха
        bx, by, bh = W - 60, 200, 260
        self.rect(bx, by, 26, bh, (0, 0, 0, 0.45))
        zone = lambda v: by + bh - clamp(v / 2, 0, 1) * bh  # noqa: E731  шкала 0..200% от идеала
        self.rect(bx, zone(1.15), 26, zone(0.85) - zone(1.15), (0.2, 0.8, 0.3, 0.5))  # идеал ±15%
        live = max(0.0, inp.velocity(now, 0.12)[1]) / ideal if inp.pinch else 0
        if live:
            self.rect(bx + 4, zone(live), 18, by + bh - zone(live), (1, 0.85, 0.2, 0.9))
        if game.last_power is not None:
            self.rect(bx - 6, zone(game.last_power) - 2, 38, 4, (1, 1, 1, 0.9))
            self.text(f"{game.last_power * 100:.0f}%", bx + 13, by + bh + 8, 18, center=True)
        self.text("сила", bx + 13, by - 30, 18, center=True)

        if mode == "camera":
            if tracker.preview:
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
                glWindowPos2d(W - PREVIEW_W - 12, 12)
                glDrawPixels(PREVIEW_W, PREVIEW_H, GL_RGB, GL_UNSIGNED_BYTE, tracker.preview)
            if not tracker.ready:
                self.text("Запуск камеры...", W / 2, H / 2, 36, center=True)
            elif not inp.visible:
                self.text("Покажите руку в камеру", W / 2, H - 120, 30, (1, 0.6, 0.3), center=True)
            hint = "Ладонь - прицел  ·  щипок - взять  ·  взмах вверх + разжать - бросок"
        else:
            hint = "Мышь - прицел  ·  зажать ЛКМ, рывок вверх и отпустить - бросок"
        self.text(hint, 16, H - 58, 20, (0.9, 0.9, 0.9))
        self.text("M - камера/мышь   R - заново   Esc - выход", 16, H - 32, 18, (0.7, 0.7, 0.7))

        if game.state == "game_over":
            self.rect(0, 0, W, H, (0, 0, 0, 0.6))
            self.text("Игра окончена", W / 2, H / 2 - 120, 56, center=True)
            self.text(f"Итог: {game.total}", W / 2, H / 2 - 40, 48, (1, 0.85, 0.2), center=True)
            if game.new_record:
                self.text("Новый рекорд!", W / 2, H / 2 + 20, 36, (0.4, 1, 0.5), center=True)
            again = "Щипок" if mode == "camera" else "Клик"
            self.text(f"{again} или ПРОБЕЛ - новая игра", W / 2, H / 2 + 80, 28, center=True)

        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)


def init_gl():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_NORMALIZE)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.22, 0.21, 0.22, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 0.93, 0.8, 1))
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.5)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.4)
    glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.3)
    glClearColor(0, 0, 0, 1)


def create_window():
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)  # 16 бит мало: мишень мерцает на фоне обода
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
    try:
        pygame.display.set_mode((W, H), pygame.DOUBLEBUF | pygame.OPENGL)
    except pygame.error:  # без сглаживания, если видеокарта не умеет MSAA
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 0)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 0)
        pygame.display.set_mode((W, H), pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Дартс 3D - управление рукой")


def main():
    pygame.init()
    create_window()
    init_gl()
    scene, hud = Scene(), Hud()
    tracker = HandTracker()
    tracker.start()
    inp, game = HandInput(), Game()
    mode = "camera"
    clock = pygame.time.Clock()

    running = True
    while running:
        dt = min(clock.tick(60) / 1000, 0.05)
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_m:
                if mode == "mouse" and tracker.error:
                    game.say(f"Камера недоступна: {tracker.error}", (1, 0.5, 0.4), 3)
                else:
                    mode = "mouse" if mode == "camera" else "camera"
                    inp.reset()
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                game.reset()
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE and game.state == "game_over":
                game.reset()

        if mode == "camera" and tracker.error:
            mode = "mouse"
            inp.reset()
            game.say(f"Камера недоступна ({tracker.error}) - играем мышью", (1, 0.5, 0.4), 4)
        if mode == "camera":
            try:
                while True:
                    inp.feed(*tracker.samples.get_nowait())
            except queue.Empty:
                pass
        else:
            mx, my = pygame.mouse.get_pos()
            inp.feed(time.time(), mx / W, my / H, pygame.mouse.get_pressed()[0], True)

        ideal = IDEAL_FLICK if mode == "camera" else MOUSE_IDEAL_FLICK
        game.update(dt, inp, ideal)
        scene.render(game, inp)
        hud.draw(game, inp, mode, tracker, ideal)
        pygame.display.flip()

    tracker.running = False
    pygame.quit()


if __name__ == "__main__":
    main()
