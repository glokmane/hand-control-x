# Copyright (c) 2026 Hand Control X authors. All rights reserved.
# Proprietary and confidential — see LICENSE. Do not redistribute.
"""
Дуэль джедаев от первого лица: вы против компьютера, управление руками через веб-камеру.
Основа: распознавание рук из hand_control.py, 3D-движок и текст из darts_game.py.

ПРАВАЯ рука - световой меч (держите кулак, как рукоять):
  * наклон кисти = наклон клинка, движение руки = движение меча;
  * БЛОК: поставьте клинок поперёк удара противника (красный призрак показывает, куда придёт удар);
  * ПАРИРОВАНИЕ: встретьте удар резким движением клинка - противник оглушён, время замедляется;
  * АТАКА: резкий взмах через тело противника, когда он рядом (после его атаки, оглушён, притянут).

ЛЕВАЯ рука - Сила:
  * ТОЛЧОК: раскрытая ладонь резко вперёд, к камере;
  * ПРИТЯЖЕНИЕ: ладонь или кулак резко назад, к себе - противник подлетает и оглушён;
  * УДУШЕНИЕ: щипок как знак OK и держать - убить не выйдет, но противник сильно замедлен;
  * ВЫРВАТЬ МЕЧ: держите раскрытую ладонь неподвижно ~0.6 с (кольцо заполнится), затем сожмите кулак.
    Лучше всего работает на оглушённом, отброшенном или душимом противнике.
    Меч прилетает в левую руку - рубите безоружного противника!

Клавиши: M - камера/мышь, X - поменять руки местами, H - призрак удара, Tab - справка,
R - заново, Esc - выход.
Мышь: меч за курсором, A/D - наклон клинка, резкое движение - удар; 1 толчок, 2 притяжение,
3 (держать) удушение, 4 вырвать меч.
"""
import math
import queue
import random
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
import pygame
from mediapipe.tasks.python import BaseOptions, vision
from OpenGL.GL import *  # noqa: F403
from OpenGL.GLU import *  # noqa: F403

from darts_game import H, PREVIEW_H, PREVIEW_W, W, Hud, create_window, make_texture
from hand_control import (CONNECTIONS, FINGERS, INDEX_TIP, MODEL_PATH, PINCH_OFF, PINCH_ON,
                          THUMB_TIP, WRIST, dist, ensure_model, extended)

# ---------------- камера и экран ----------------
EYE = (0.0, 1.6, 0.0)          # глаза игрока, смотрим вдоль -Z
FOV = 60
ASPECT = W / H
TAN = math.tan(math.radians(FOV / 2))
NEAR_CLIP = 0.12
CAM_ASPECT = 4 / 3             # кадр веб-камеры 640x480

# ---------------- мечи ----------------
BLADE_LEN, HILT_FRONT, HILT_BACK = 1.0, 0.13, 0.12
PLAYER_COLOR, ENEMY_COLOR = (0.3, 0.6, 1.0), (1.0, 0.12, 0.08)
SWING_SPEED = 2.0              # скорость клинка на экране (экранов/с), с которой взмах считается ударом
PARRY_SPEED = 1.4              # скорость клинка при блоке, превращающая его в парирование
PLAYER_REACH = 2.1             # дальше этого противника мечом не достать

# ---------------- противник ----------------
IDLE_DIST, STRIKE_DIST, PULL_DIST = 2.3, 1.6, 1.35
WINDUP, COMBO_WINDUP = 0.8, 0.5
ENEMY_HP, PLAYER_HP = 100, 100
HIT_DAMAGE, ENEMY_HIT_DAMAGE, FORCE_PUSH_DAMAGE = 10, 12, 8
BLOCK_CHANCE = {"idle": 0.8, "approach": 0.8, "retreat": 0.7, "strafe": 0.7, "block": 0.8,
                "windup": 0.35, "force_windup": 0.2, "recoil": 0.3}

# ---------------- Сила ----------------
PUSH_COST, PULL_COST, DISARM_COST, CHOKE_COST = 30, 25, 40, 20   # CHOKE_COST - в секунду
FORCE_REGEN = 9
PUSH_RATIO, PULL_RATIO = 1.25, 0.75    # во сколько раз должна измениться видимая ладонь за 0.3 с
DISARM_CHARGE = 0.6


# ======================= векторы =======================

def v_add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def v_sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def v_mul(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def v_dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def v_len(a): return math.sqrt(v_dot(a, a))
def v_lerp(a, b, k): return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k)


def v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def v_norm(a):
    n = v_len(a)
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-9 else (0.0, 1.0, 0.0)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def ease(k):
    k = clamp(k, 0, 1)
    return k * k * (3 - 2 * k)


def slerp(a, b, k):
    d = clamp(v_dot(a, b), -1, 1)
    if d > 0.9995:
        return v_norm(v_lerp(a, b, k))
    th = math.acos(d)
    s = math.sin(th)
    return v_add(v_mul(a, math.sin((1 - k) * th) / s), v_mul(b, math.sin(k * th) / s))


def lerp_pose(p, q, k):
    return v_lerp(p[0], q[0], k), slerp(p[1], q[1], k)


def keyframes_at(frames, p):
    n = len(frames) - 1
    seg = min(int(p * n), n - 1)
    return lerp_pose(frames[seg], frames[seg + 1], ease(p * n - seg))


# ======================= экранная геометрия (для блоков и попаданий) =======================

def project(p):
    x, y, z = v_sub(p, EYE)
    d = max(-z, 1e-3)
    return (x / d / (TAN * ASPECT), y / d / TAN)


def clip_project(a, b):
    """Отрезок мира -> отрезок на экране (обрезая часть за ближней плоскостью)."""
    za, zb = a[2] - EYE[2], b[2] - EYE[2]
    if za > -NEAR_CLIP and zb > -NEAR_CLIP:
        return None
    if za > -NEAR_CLIP:
        a = v_lerp(a, b, (za + NEAR_CLIP) / (za - zb))
    elif zb > -NEAR_CLIP:
        b = v_lerp(b, a, (zb + NEAR_CLIP) / (zb - za))
    return project(a), project(b), a, b


def seg_intersect(p1, p2, q1, q2):
    r = (p2[0] - p1[0], p2[1] - p1[1])
    s = (q2[0] - q1[0], q2[1] - q1[1])
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-9:
        return None
    qp = (q1[0] - p1[0], q1[1] - p1[1])
    t = (qp[0] * s[1] - qp[1] * s[0]) / den
    u = (qp[0] * r[1] - qp[1] * r[0]) / den
    return (t, u) if 0 <= t <= 1 and 0 <= u <= 1 else None


def seg_rect(a, b, rect):
    x0, y0, x1, y1 = rect
    inside = lambda p: x0 <= p[0] <= x1 and y0 <= p[1] <= y1  # noqa: E731
    if inside(a) or inside(b):
        return True
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return any(seg_intersect(a, b, corners[i], corners[(i + 1) % 4]) for i in range(4))


# ======================= руки: камера -> мир =======================

def lm_xyz(p):
    return (p.x * CAM_ASPECT, p.y, p.z * CAM_ASPECT)


class HandFilter:
    """Сглаженные ориентиры одной руки и их перенос в 3D перед камерой."""
    HAND_SIZE = 0.085          # длина запястье -> основание среднего пальца в игре, м

    def __init__(self, side):
        self.side = side       # +1 правая, -1 левая
        self.pts = None
        self.last_t = 0.0

    def feed(self, t, lm):
        if lm is None:
            if t - self.last_t > 0.35:
                self.pts = None
            return
        raw = [lm_xyz(p) for p in lm]
        if self.pts is None:
            self.pts = raw
        else:
            # быстрые движения почти без сглаживания (меньше задержка), медленные - с сильным
            move = math.hypot(raw[0][0] - self.pts[0][0], raw[0][1] - self.pts[0][1])
            a = clamp(move * 12, 0.35, 0.9)
            self.pts = [v_lerp(p, r, a) for p, r in zip(self.pts, raw)]
        self.last_t = t

    def world(self):
        """-> (21 точка в мире, центр кисти, направление клинка) или None."""
        if self.pts is None:
            return None
        pts = self.pts
        c = v_mul(v_add(v_add(v_add(pts[0], pts[5]), v_add(pts[9], pts[13])), pts[17]), 0.2)
        s = self.HAND_SIZE / max(v_len(v_sub(pts[9], pts[0])), 1e-4)
        anchor = (EYE[0] + (c[0] / CAM_ASPECT - 0.5) * 1.25, EYE[1] - 0.1 - (c[1] - 0.5) * 0.65, -0.55)
        world = [(anchor[0] + (p[0] - c[0]) * s, anchor[1] - (p[1] - c[1]) * s, anchor[2] - (p[2] - c[2]) * s)
                 for p in pts]
        dx, dy = world[9][0] - world[0][0], world[9][1] - world[0][1]
        n = math.hypot(dx, dy) or 1
        blade = v_norm((dx / n * 0.85, dy / n * 0.85, -0.5))   # клинок наклонён от игрока к противнику
        return world, anchor, blade


class ForceGestures:
    """Жесты левой руки: толчок, притяжение, удушение, вырывание меча."""

    def __init__(self):
        self.hist = deque(maxlen=60)
        self.pose = None
        self._raw, self._count = None, 0
        self.pinch = False
        self.charge = 0.0
        self.charged_at = None
        self.cooldown_until = 0.0
        self.last_t = None

    def reset(self):
        self.hist.clear()
        self.pose, self.charge, self.charged_at, self.last_t = None, 0.0, None, None

    def feed(self, t, lm):
        if lm is None:
            self.reset()
            return None
        dt = 0 if self.last_t is None else t - self.last_t
        self.last_t = t
        palm = dist(lm[WRIST], lm[9]) or 1e-6
        up = {f: extended(lm, f) for f in FINGERS}
        n_up = sum(up.values())
        pd = dist(lm[THUMB_TIP], lm[INDEX_TIP]) / palm
        # щипок как знак OK: остальные пальцы не сжаты - иначе это кулак
        self.pinch = pd < (PINCH_OFF if self.pinch else PINCH_ON) and (up["middle"] or up["ring"] or up["pinky"])
        raw = "pinch" if self.pinch else "open" if n_up >= 4 else "fist" if n_up == 0 else "other"
        self._count = self._count + 1 if raw == self._raw else 1
        self._raw = raw
        if self._count >= 2:             # поза должна продержаться 2 кадра - меньше ложных срабатываний
            self.pose = raw

        p0, p5, p17 = lm_xyz(lm[0]), lm_xyz(lm[5]), lm_xyz(lm[17])
        size = v_len(v_sub(p0, p5)) + v_len(v_sub(p5, p17)) + v_len(v_sub(p17, p0))
        cx, cy = lm[9].x, lm[9].y
        self.hist.append((t, size, cx, cy))
        if t < self.cooldown_until:
            return None
        recent = [h for h in self.hist if t - h[0] <= 0.3]
        sizes = [h[1] for h in recent]
        event = None
        if self.pose == "open" and size / min(sizes) >= PUSH_RATIO:
            event = "push"
        elif self.pose == "fist" and self.charged_at is not None and t - self.charged_at < 1.0:
            event = "disarm"
        elif self.pose in ("open", "fist") and size / max(sizes) <= PULL_RATIO:
            event = "pull"
        else:
            old = [h for h in self.hist if t - h[0] <= 0.15]
            speed = math.hypot(cx - old[0][2], cy - old[0][3]) / max(t - old[0][0], 1e-3) if old else 0
            if self.pose == "open" and speed < 0.4:
                self.charge = min(DISARM_CHARGE, self.charge + dt)
                if self.charge >= DISARM_CHARGE:
                    self.charged_at = t
            elif self.pose != "fist":
                self.charge = 0.0
        if event:
            self.cooldown_until = t + 0.9
            self.charge, self.charged_at = 0.0, None
            self.hist.clear()
        return event


class TwoHandTracker(threading.Thread):
    """Фоновый поток: камера -> MediaPipe (2 руки) -> (t, левая, правая)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = queue.SimpleQueue()
        self.preview = None
        self.error = None
        self.ready = False
        self.running = True
        self.swap = False

    def run(self):
        try:
            ensure_model()
            options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
                running_mode=vision.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not cap.isOpened():
                self.error = "веб-камера не найдена"
                return
            with vision.HandLandmarker.create_from_options(options) as landmarker:
                self._loop(cap, landmarker)
            cap.release()
        except Exception as e:
            self.error = str(e)

    def _loop(self, cap, landmarker):
        last_ts, t0 = -1, time.time()
        while self.running:
            ok, frame = cap.read()
            if not ok:
                continue
            t = time.time()
            frame = cv2.flip(frame, 1)
            ts = max(int((t - t0) * 1000), last_ts + 1)
            last_ts = ts
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
            hands = {"L": None, "R": None}
            lms = res.hand_landmarks
            labels = [h[0].category_name for h in res.handedness]
            if len(lms) == 2 and labels[0] == labels[1]:
                # модель назвала обе руки одинаково - разводим по положению в кадре
                a, b = sorted(lms, key=lambda lm: lm[0].x)
                hands = {"L": a, "R": b}
            else:
                # кадр зеркальный (как селфи) - именно так MediaPipe и определяет руку
                for lm, lab in zip(lms, labels):
                    hands["R" if lab == "Right" else "L"] = lm
            if self.swap:
                hands = {"L": hands["R"], "R": hands["L"]}
            self.samples.put((t, hands["L"], hands["R"]))
            self.preview = self._make_preview(frame, hands)
            self.ready = True

    @staticmethod
    def _make_preview(frame, hands):
        small = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
        for key, color in (("R", (255, 160, 60)), ("L", (60, 220, 255))):
            lm = hands[key]
            if lm:
                pts = [(int(p.x * PREVIEW_W), int(p.y * PREVIEW_H)) for p in lm]
                for a, b in CONNECTIONS:
                    cv2.line(small, pts[a], pts[b], color, 2)
                cv2.putText(small, key, (pts[0][0] - 6, pts[0][1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return cv2.flip(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), 0).tobytes()


# ======================= световой меч игрока =======================

class Saber:
    def __init__(self, color, hilt, direction):
        self.color = color
        self.hilt, self.dir = hilt, direction
        self.track = deque(maxlen=8)      # (t, точка клинка на экране) - для скорости взмаха
        self.trail = deque(maxlen=9)      # след клинка
        self.speed = 0.0
        self.next_hit = 0.0

    def base(self):
        return v_add(self.hilt, v_mul(self.dir, HILT_FRONT))

    def tip(self):
        return v_add(self.hilt, v_mul(self.dir, HILT_FRONT + BLADE_LEN))

    def move_to(self, hilt, direction, dt):
        k = min(1.0, dt * 25)
        self.hilt = v_lerp(self.hilt, hilt, k)
        self.dir = v_norm(v_lerp(self.dir, direction, k))

    def update_motion(self, now):
        p = project(v_lerp(self.base(), self.tip(), 0.6))
        self.track.append((now, p))
        old = next((s for s in self.track if now - s[0] <= 0.07), self.track[0])
        dt = now - old[0]
        self.speed = math.hypot(p[0] - old[1][0], p[1] - old[1][1]) / dt if dt > 1e-3 else 0.0
        self.trail.append((self.base(), self.tip()))


REST_R = ((0.22, 1.33, -0.5), v_norm((-0.1, 0.45, -0.85)))
REST_L = ((-0.24, 1.33, -0.5), v_norm((0.1, 0.45, -0.85)))
KNOCKED_R = ((0.35, 1.1, -0.45), v_norm((0.8, -0.2, -0.5)))


class Player:
    def __init__(self):
        self.hp = PLAYER_HP
        self.force = 100.0
        self.right = Saber(PLAYER_COLOR, *REST_R)
        self.left_saber = None            # трофейный меч противника в левой руке
        self.knocked_until = 0.0
        self.choking = False
        self.right_view = self.left_view = None   # (точки, центр, клинок) для рисования
        self.left_anchor = REST_L[0]

    def sabers(self):
        return [s for s in (self.right, self.left_saber) if s]

    def spend(self, cost):
        if self.force < cost:
            return False
        self.force -= cost
        return True


# ======================= противник =======================
# Позы меча противника: (кисть, направление клинка) в его системе координат:
# x - его правая сторона, y - вверх, z - вперёд (к игроку). Корень - между ступнями.

def P(h, d):
    return (h, v_norm(d))


def mirror(frames):
    return [((-h[0], h[1], h[2]), (-d[0], d[1], d[2])) for h, d in frames]


GUARD = P((0.12, 1.12, 0.32), (-0.25, 0.8, 0.55))
RECOIL = P((0.22, 1.35, 0.12), (0.3, 0.9, -0.2))
STAGGER = P((0.38, 1.25, -0.05), (0.7, 0.6, -0.4))
_DIAG = [P((0.4, 1.7, -0.05), (0.55, 0.6, -0.58)), P((0.3, 1.65, 0.25), (0.5, 0.8, 0.33)),
         P((0.1, 1.4, 0.5), (-0.25, -0.05, 0.97)), P((-0.2, 0.95, 0.45), (-0.75, -0.5, 0.43))]
_HORIZ = [P((0.45, 1.3, -0.05), (0.75, 0.15, -0.64)), P((0.35, 1.3, 0.3), (0.7, 0.12, 0.7)),
          P((0.05, 1.3, 0.5), (0, 0.08, 1)), P((-0.3, 1.25, 0.4), (-0.75, 0.05, 0.66))]
_LOW = [P((0.4, 1.0, -0.05), (0.7, -0.25, -0.67)), P((0.3, 0.9, 0.3), (0.75, -0.3, 0.6)),
        P((0.05, 0.85, 0.5), (0, -0.35, 0.94)), P((-0.3, 0.85, 0.4), (-0.7, -0.35, 0.62))]
ATTACKS = {   # имя: (ключевые позы, индекс позы, в которой клинок достигает игрока)
    "удар сверху": ([P((0.05, 1.85, -0.05), (0, 0.45, -0.89)), P((0.04, 1.75, 0.25), (0, 0.95, 0.3)),
                P((0.03, 1.5, 0.5), (0, 0.1, 0.99)), P((0.0, 1.05, 0.5), (0, -0.65, 0.76))], 2),
    # его правая сторона - слева на экране, поэтому названия - с точки зрения игрока
    "удар наискось слева": (_DIAG, 2),
    "удар наискось справа": (mirror(_DIAG), 2),
    "удар слева": (_HORIZ, 2),
    "удар справа": (mirror(_HORIZ), 2),
    "укол": ([P((0.22, 1.15, -0.2), (0, 0.05, 1)), P((0.08, 1.4, 0.75), (-0.05, 0.12, 0.99)),
              P((0.08, 1.4, 0.75), (-0.05, 0.12, 0.99))], 1),
    "удар снизу": (_LOW, 2),
}


def ik(s, t, a, b, pole):
    """Двухзвенная рука: плечо s, цель t -> (локоть, кисть)."""
    d = v_sub(t, s)
    L = clamp(v_len(d), 1e-4, a + b - 1e-3)
    dn = v_norm(d)
    cos_a = clamp((a * a + L * L - b * b) / (2 * a * L), -1, 1)
    perp = v_norm(v_sub(pole, v_mul(dn, v_dot(pole, dn))))
    elbow = v_add(s, v_add(v_mul(dn, a * cos_a), v_mul(perp, a * math.sqrt(1 - cos_a * cos_a))))
    return elbow, v_add(s, v_mul(dn, L))


class Enemy:
    def __init__(self, game):
        self.game = game
        self.x, self.z = 0.0, -3.2
        self.hp = ENEMY_HP
        self.has_saber = True
        self.state, self.t = "idle", 0.0
        self.idle_time = 3.0                 # пауза перед боем
        self.from_pose = GUARD
        self.choked = False
        self.lift = 0.0
        self.walk = 0.0
        self.lean = 0.0
        self.combo_left = 0
        self.attack = None
        self.last_attack = None
        self.push_ready_at = 6.0
        self.stagger_dur = 1.0
        self.kv = 0.0
        self.block_pose = GUARD
        self.resolved = False
        self.trail = deque(maxlen=9)

    # --- геометрия ---
    def dist(self):
        return math.hypot(self.x, self.z)

    def yaw(self):
        return math.atan2(-self.x, -self.z)    # лицом к игроку

    def to_world(self, p):
        c, s = math.cos(self.yaw()), math.sin(self.yaw())
        lx, ly, lz = -p[0], p[1], p[2]       # его правая сторона - слева на экране
        return (self.x + c * lx + s * lz, ly + self.lift, self.z - s * lx + c * lz)

    def dir_world(self, d):
        c, s = math.cos(self.yaw()), math.sin(self.yaw())
        lx, ly, lz = -d[0], d[1], d[2]
        return (c * lx + s * lz, ly, -s * lx + c * lz)

    def move_to_dist(self, target, speed, dt):
        d = self.dist()
        step = clamp(target - d, -speed * dt, speed * dt)
        nd = clamp(d + step, 1.2, 5.5)
        self.x, self.z = self.x / d * nd, self.z / d * nd
        self.walk += abs(step) * 9
        return abs(target - nd) < 0.03

    # --- поза ---
    def target_pose(self):
        s = self.state
        if s == "windup":
            return self.frames[0], self.windup_dur
        if s == "strike":
            return keyframes_at(self.frames, min(1.0, self.t / self.strike_dur)), 0
        if s == "recover":
            return GUARD, 0.5
        if s == "recoil":
            return RECOIL, 0.12
        if s in ("staggered", "knocked", "pulled"):
            return STAGGER, 0.15
        if s == "block":
            return self.block_pose, 0.08
        now = self.game.clock
        h = GUARD[0]
        return ((h[0] + math.sin(now * 1.3) * 0.03, h[1] + math.sin(now * 2.1) * 0.02, h[2]), GUARD[1]), 0.3

    def saber_pose(self):
        target, blend = self.target_pose()
        if blend <= 0:
            return target
        return lerp_pose(self.from_pose, target, ease(self.t / blend))

    def left_hand_target(self):
        if self.choked:
            return (-0.04, 1.5, 0.13)
        if self.state in ("force_windup", "recall"):
            return v_lerp((-0.22, 1.0, 0.18), (-0.1, 1.45, 0.62), ease(self.t / 0.3))
        return (-0.22, 1.0 + math.sin(self.game.clock * 1.7) * 0.02, 0.18)

    def skeleton(self):
        """Суставы в его системе координат с учётом наклона корпуса."""
        hip = (0.0, 0.95, 0.0)
        a = math.radians(self.lean)

        def lean(p):
            r = v_sub(p, hip)
            return v_add(hip, (r[0], r[1] * math.cos(a) - r[2] * math.sin(a), r[1] * math.sin(a) + r[2] * math.cos(a)))

        sk = {"hip": hip, "neck": lean((0, 1.5, 0)), "head": lean((0, 1.66, 0.02)),
              "shR": lean((0.2, 1.45, 0)), "shL": lean((-0.2, 1.45, 0)), "lean": lean}
        hand, d = self.saber_pose() if self.has_saber else (((0.25, 1.0, 0.2), (0, 1, 0)))
        sk["elbowR"], sk["handR"] = ik(sk["shR"], hand, 0.34, 0.34, (0.5, -0.6, -0.3))
        sk["elbowL"], sk["handL"] = ik(sk["shL"], self.left_hand_target(), 0.34, 0.34, (-0.5, -0.6, -0.3))
        sk["blade"] = d
        return sk

    def saber_world(self, sk=None):
        if not self.has_saber or self.state == "dead":
            return None
        sk = sk or self.skeleton()
        hand, d = self.to_world(sk["handR"]), self.dir_world(sk["blade"])
        return v_add(hand, v_mul(d, HILT_FRONT)), v_add(hand, v_mul(d, HILT_FRONT + BLADE_LEN))

    def chest_world(self):
        return self.to_world((0, 1.3, 0.05))

    def hitbox(self):
        sk = self.skeleton()
        pts = [self.to_world(p) for p in (sk["head"], sk["neck"], sk["shR"], sk["shL"], (0.2, 0.9, 0), (-0.2, 0.9, 0))]
        scr = [project(p) for p in pts]
        xs, ys = [p[0] for p in scr], [p[1] for p in scr]
        return min(xs), min(ys), max(xs), max(ys) + 0.04

    # --- состояния ---
    def set_state(self, s):
        if not self.has_saber and s in ("idle", "approach", "retreat", "strafe", "recover", "windup", "force_windup"):
            s = "disarmed"                     # без меча не нападает - пятится и ждёт, чтобы вернуть его
        self.from_pose = self.saber_pose() if self.has_saber else GUARD
        self.state, self.t = s, 0.0

    def start_windup(self, dur):
        name = random.choice([n for n in ATTACKS if n != self.last_attack])
        self.attack = self.last_attack = name
        self.frames, impact = ATTACKS[name]
        n = len(self.frames) - 1
        impact_p = impact / n
        self.block_lo, self.hit_p = max(0.0, impact_p - 0.12), impact_p + 0.04
        self.windup_dur = dur
        self.strike_dur = 0.12 * n + 0.1
        self.set_state("windup")

    def decide(self):
        if self.dist() > IDLE_DIST + 0.15:
            return self.set_state("approach")
        r = random.random()
        if r < 0.62:
            self.combo_left = random.choice([1, 1, 2, 2, 3])
            self.start_windup(WINDUP)
        elif r < 0.77 and self.game.clock >= self.push_ready_at:
            self.push_ready_at = self.game.clock + 7
            self.set_state("force_windup")
        else:
            self.strafe_dir = -1 if self.x > 0.5 else 1 if self.x < -0.5 else random.choice([-1, 1])
            self.strafe_dur = random.uniform(0.6, 1.1)
            self.set_state("strafe")

    def idle_for(self, secs):
        self.idle_time = secs
        self.set_state("idle")

    def update(self, dt):
        if self.choked:
            dt *= 0.35                     # удушение не убивает, но сильно замедляет
        self.lift += ((0.18 if self.choked else 0.0) - self.lift) * min(1.0, dt * 8)
        target_lean = {"strike": 15, "windup": -5, "knocked": -35, "staggered": -15, "pulled": 10,
                       "recoil": -8, "dead": 0}.get(self.state, 0) - (10 if self.choked else 0)
        self.lean += (target_lean - self.lean) * min(1.0, dt * 10)
        if self.state == "strike":
            self.strike_step(dt)
        else:
            self.t += dt
            getattr(self, "st_" + self.state)(dt)
        sw = self.saber_world()
        if sw:
            self.trail.append(sw)
        else:
            self.trail.clear()

    def strike_step(self, dt):
        for _ in range(4):              # подшаги - быстрый клинок не "проскакивает" сквозь блок
            self.t += dt / 4
            p = self.t / self.strike_dur
            if not self.resolved:
                if self.block_lo <= p <= self.hit_p:
                    parry = self.game.check_block()
                    if parry is not None:
                        return self.on_blocked(parry)
                if p >= self.hit_p:
                    self.resolved = True
                    self.game.enemy_hits_player()
                    if self.state != "strike":
                        return
            if p >= 1:
                return self.after_strike()

    def after_strike(self):
        self.combo_left -= 1
        if self.combo_left > 0:
            self.start_windup(COMBO_WINDUP)
        else:
            self.set_state("recover")

    def on_blocked(self, parry):
        if parry:
            self.stagger_dur = 1.5
            self.set_state("staggered")
            self.game.slowmo(0.8)
            self.game.say("Парирование!", (0.5, 0.9, 1))
        else:
            self.set_state("recoil")

    def st_idle(self, dt):
        if self.t > self.idle_time:
            self.decide()

    def st_victory(self, dt):
        pass

    def st_approach(self, dt):
        if self.move_to_dist(IDLE_DIST, 1.8, dt):
            self.idle_for(random.uniform(0.3, 0.7))

    def st_retreat(self, dt):
        if self.move_to_dist(IDLE_DIST, 1.6, dt):
            self.idle_for(random.uniform(0.3, 0.8))

    def st_strafe(self, dt):
        d = self.dist()
        self.x = clamp(self.x + self.strafe_dir * 1.1 * dt, -0.9, 0.9)
        self.z = -math.sqrt(max(d * d - self.x * self.x, 1.0))
        self.walk += 1.1 * dt * 9
        if self.t > self.strafe_dur:
            self.idle_for(0.2)

    def st_windup(self, dt):
        left = max(self.windup_dur - self.t, 0.05)
        self.move_to_dist(STRIKE_DIST, max(0.0, (self.dist() - STRIKE_DIST) / left), dt)
        if self.t >= self.windup_dur:
            self.resolved = False
            self.set_state("strike")
            self.from_pose = self.frames[0]

    def st_recover(self, dt):
        if self.t > 0.8:
            self.set_state("retreat")

    def st_recoil(self, dt):
        if self.t > 0.35:
            self.combo_left -= 1
            if self.combo_left > 0:
                self.start_windup(COMBO_WINDUP)
            else:
                self.set_state("retreat")

    def st_staggered(self, dt):
        if self.t > self.stagger_dur:
            self.set_state("retreat")

    def st_knocked(self, dt):
        d = self.dist()
        nd = min(5.5, d + self.kv * dt)
        self.x, self.z = self.x / d * nd, self.z / d * nd
        self.kv = max(0.0, self.kv - 9 * dt)
        if self.t > 1.1:
            self.set_state("approach")

    def st_pulled(self, dt):
        if self.move_to_dist(PULL_DIST, 9, dt) or self.t > 0.6:
            self.stagger_dur = 1.3
            self.set_state("staggered")

    def st_block(self, dt):
        if self.t > 0.25:
            if random.random() < 0.35:
                self.combo_left = 1
                self.start_windup(COMBO_WINDUP)     # ответный удар
            else:
                self.idle_for(0.3)

    def st_force_windup(self, dt):
        if self.t > 0.85:
            self.game.enemy_force_push()
            self.combo_left = 1
            self.start_windup(COMBO_WINDUP)

    def st_disarmed(self, dt):
        self.move_to_dist(3.0, 1.2, dt)
        if self.game.clock - self.disarmed_at > 6 and not self.game.flying:
            self.set_state("recall")
            self.game.say("Он притягивает меч обратно - руби!", (1, 0.6, 0.3), 1.3)

    def st_recall(self, dt):
        if self.t > 1.3:
            self.game.recall_saber()
            self.disarmed_at = self.game.clock    # пока меч летит обратно, он всё ещё безоружен

    def st_dead(self, dt):
        pass

    # --- реакции ---
    def knock_back(self, speed):
        self.kv = speed
        self.set_state("knocked")

    def on_player_swing(self):
        """Игрок попал клинком в противника: блок или урон."""
        if not self.has_saber:
            return self.game.kill_enemy(disarmed=True)
        chance = BLOCK_CHANCE.get(self.state, 0.0) * (0.4 if self.choked else 1.0)
        if self.has_saber and random.random() < chance:
            self.block_against(self.game.player)
            return "block"
        vulnerable = self.state in ("staggered", "knocked", "pulled", "recover")
        self.hp -= HIT_DAMAGE * (1.8 if vulnerable else 1.0)
        if self.hp <= 0:
            return self.game.kill_enemy(disarmed=False)
        if not vulnerable:
            self.stagger_dur = 0.45
            self.set_state("staggered")
        return "hit"

    def block_against(self, player):
        s = max(player.sabers(), key=lambda sb: sb.speed)
        a, b = project(s.base()), project(s.tip())
        th = math.atan2(b[1] - a[1], b[0] - a[0]) + math.pi / 2
        d = (-math.cos(th), math.sin(th), 0.35)
        if d[1] < 0:
            d = (-d[0], -d[1], d[2])
        mid_x = (a[0] + b[0]) / 2
        self.block_pose = ((-clamp(mid_x * 0.4, -0.3, 0.3), 1.3, 0.4), v_norm(d))
        self.set_state("block")

    def lose_saber(self):
        self.has_saber = False
        self.disarmed_at = self.game.clock
        self.set_state("disarmed")

    def regain_saber(self):
        self.has_saber = True
        self.from_pose = GUARD


# ======================= эффекты =======================

class Effects:
    def __init__(self):
        self.sparks = []          # [pos, vel, life, color]
        self.waves = []           # {a, b, t, dur, color}
        self.flash = None         # (pos, сила, цвет)
        self.damage = 0.0
        self.shake = 0.0

    def burst(self, pos, n=30, color=(1, 0.75, 0.3), speed=2.5):
        for _ in range(n):
            v = v_mul(v_norm((random.uniform(-1, 1), random.uniform(-0.5, 1.2), random.uniform(-1, 1))),
                      random.uniform(0.4, 1) * speed)
            self.sparks.append([pos, v, random.uniform(0.2, 0.5), color])
        self.flash = (pos, 1.0, color)

    def wave(self, a, b, color=(0.6, 0.8, 1.0), dur=0.25):
        self.waves.append({"a": a, "b": b, "t": 0.0, "dur": dur, "color": color})

    def update(self, dt):
        for s in self.sparks:
            s[1] = (s[1][0], s[1][1] - 6 * dt, s[1][2])
            s[0] = v_add(s[0], v_mul(s[1], dt))
            s[2] -= dt
        self.sparks = [s for s in self.sparks if s[2] > 0]
        for w in self.waves:
            w["t"] += dt
        self.waves = [w for w in self.waves if w["t"] < w["dur"] + 0.15]
        if self.flash:
            p, k, c = self.flash
            self.flash = (p, k - dt * 5, c) if k > dt * 5 else None
        self.damage = max(0.0, self.damage - dt * 1.5)
        self.shake = max(0.0, self.shake - dt * 2)


# ======================= игра =======================

class Game:
    def __init__(self):
        self.reset()

    def reset(self):
        self.clock = 0.0
        self.player = Player()
        self.enemy = Enemy(self)
        self.fx = Effects()
        self.state = "fight"
        self.slow_until = 0.0
        self.time_scale = 1.0
        self.pending = []           # отложенные действия: (время, функция)
        self.flying = None          # летящий меч
        self.message = ("Бой!", (1, 1, 1), time.time() + 2)
        self.choke_told = False

    def say(self, text, color=(1, 1, 1), secs=1.6):
        self.message = (text, color, time.time() + secs)

    def slowmo(self, secs):
        self.slow_until = time.time() + secs

    def later(self, delay, fn):
        self.pending.append((self.clock + delay, fn))

    # --- главный шаг ---
    def update(self, dt_real, hands):
        now = time.time()
        self.time_scale = 0.3 if now < self.slow_until else 1.0
        dt = dt_real * self.time_scale
        self.clock += dt
        p, e = self.player, self.enemy

        self.apply_hands(hands, dt_real, now)
        for s in p.sabers():
            s.update_motion(now)

        # удушение
        p.choking = (hands["choke"] and self.state == "fight" and e.state not in ("dead",)
                     and p.left_saber is None and p.force > 0)
        if p.choking:
            p.force = max(0.0, p.force - CHOKE_COST * dt_real)
            if not self.choke_told:
                self.say("Удушение его не берёт, но он замедлен", (0.6, 0.85, 1), 2)
                self.choke_told = True
        else:
            p.force = min(100.0, p.force + FORCE_REGEN * dt_real)
        e.choked = p.choking

        for ev in hands["events"]:
            if self.state == "fight":
                {"push": self.force_push, "pull": self.force_pull, "disarm": self.force_disarm}[ev]()

        due = [f for f in self.pending if f[0] <= self.clock]
        self.pending = [f for f in self.pending if f[0] > self.clock]
        for _, fn in due:
            fn()

        e.update(dt)
        self.update_flying(dt)
        if self.state == "fight":
            self.check_swings(now)
        self.fx.update(dt)

    def apply_hands(self, hands, dt, now):
        p = self.player
        p.right_view, p.left_view = hands["right"], hands["left"]
        if now < p.knocked_until:
            p.right.move_to(*KNOCKED_R, dt)
        elif hands["right"]:
            _, anchor, blade = hands["right"]
            p.right.move_to(anchor, blade, dt)
        elif hands["right_pose"]:
            p.right.move_to(*hands["right_pose"], dt)
        else:
            p.right.move_to(*REST_R, dt)
        if hands["left"]:
            p.left_anchor = hands["left"][1]
        else:
            p.left_anchor = REST_L[0]
        if p.left_saber:
            if hands["left"]:
                p.left_saber.move_to(hands["left"][1], hands["left"][2], dt)
            else:
                p.left_saber.move_to(*REST_L, dt)

    # --- бой на мечах ---
    def check_block(self):
        """Вызывается во время удара противника: пересекает ли его клинок наш (на экране)."""
        p = self.player
        if time.time() < p.knocked_until:
            return None
        sw = self.enemy.saber_world()
        seg_e = sw and clip_project(*sw)
        if not seg_e:
            return None
        for s in p.sabers():
            seg_p = clip_project(s.base(), s.tip())
            if not seg_p:
                continue
            hit = seg_intersect(seg_p[0], seg_p[1], seg_e[0], seg_e[1])
            if hit:
                point = v_lerp(seg_p[2], seg_p[3], hit[0])
                parry = s.speed > PARRY_SPEED
                self.fx.burst(point, 45 if parry else 25, (1, 0.8, 0.4), 3.0 if parry else 2.0)
                return parry
        return None

    def enemy_hits_player(self):
        p = self.player
        p.hp -= ENEMY_HIT_DAMAGE
        self.fx.damage, self.fx.shake = 1.0, 0.6
        self.say("Пропущен удар!", (1, 0.35, 0.3), 1.0)
        if p.hp <= 0:
            self.lose()

    def check_swings(self, now):
        e = self.enemy
        if e.state == "dead" or e.dist() > PLAYER_REACH:
            return
        box = None
        for s in self.player.sabers():
            if s.speed < SWING_SPEED or now < s.next_hit:
                continue
            seg = clip_project(s.base(), s.tip())
            box = box or e.hitbox()
            if seg and seg_rect(seg[0], seg[1], box):
                s.next_hit = now + 0.45
                result = e.on_player_swing()
                chest = e.chest_world()
                if result == "block":
                    self.fx.burst(v_lerp(chest, EYE, 0.25), 20)
                elif result == "hit":
                    self.fx.burst(chest, 25, (1, 0.45, 0.15), 1.5)
                    self.say("Попадание!", (1, 0.85, 0.3), 0.8)
                return

    # --- Сила ---
    def force_push(self):
        p, e = self.player, self.enemy
        if not p.spend(PUSH_COST):
            return self.say("Не хватает Силы", (0.6, 0.7, 1), 1)
        self.fx.wave(p.left_anchor, e.chest_world())
        if e.state == "force_windup":
            e.knock_back(3.0)
            self.fx.burst(v_lerp(p.left_anchor, e.chest_world(), 0.5), 40, (0.6, 0.8, 1), 3)
            self.fx.shake = 0.4
            return self.say("Столкновение Силы!", (0.6, 0.85, 1))

        def hit():
            if e.state in ("dead", "victory"):
                return
            e.hp -= 5
            e.knock_back(7.0)
            if e.hp <= 0:
                self.kill_enemy(False)
        self.later(0.15, hit)
        self.say("Толчок Силы", (0.6, 0.85, 1), 0.8)

    def force_pull(self):
        p, e = self.player, self.enemy
        if e.state in ("dead", "victory"):
            return
        if not p.spend(PULL_COST):
            return self.say("Не хватает Силы", (0.6, 0.7, 1), 1)
        self.fx.wave(e.chest_world(), p.left_anchor, (0.7, 0.75, 1))
        guarding = e.state in ("idle", "approach", "retreat", "strafe", "block")
        if guarding and not e.choked and random.random() < 0.2:
            return self.say("Он устоял", (0.8, 0.8, 0.8))
        e.set_state("pulled")
        self.say("Притяжение!", (0.6, 0.85, 1), 0.8)

    def force_disarm(self):
        p, e = self.player, self.enemy
        if p.left_saber or not e.has_saber or e.state in ("dead", "victory"):
            return
        if not p.spend(DISARM_COST):
            return self.say("Не хватает Силы", (0.6, 0.7, 1), 1)
        sw = e.saber_world()
        self.fx.wave(p.left_anchor, sw[0], (0.8, 0.7, 1), 0.2)
        chance = 0.85 if e.choked else {"staggered": 0.9, "knocked": 0.9, "pulled": 0.9, "recover": 0.6,
                                        "recoil": 0.6, "windup": 0.35}.get(e.state, 0.15)
        if random.random() > chance:
            self.say("Он удержал меч", (0.8, 0.8, 0.8))
            if e.state in ("idle", "strafe", "approach"):
                e.combo_left = 1
                e.start_windup(COMBO_WINDUP)
            return
        hand_dir = e.dir_world(e.skeleton()["blade"])
        e.lose_saber()
        self.flying = {"from": e.to_world(e.skeleton()["handR"]), "dir": hand_dir, "t": 0.0, "dur": 0.5,
                       "to": "player", "pos": sw[0]}
        self.say("Меч вырван! Руби его!", (1, 0.85, 0.3), 2)

    def recall_saber(self):
        p = self.player
        if p.left_saber is None:
            return
        self.flying = {"from": p.left_saber.hilt, "dir": p.left_saber.dir, "t": 0.0, "dur": 0.5,
                       "to": "enemy", "pos": p.left_saber.hilt}
        p.left_saber = None

    def update_flying(self, dt):
        f = self.flying
        if not f:
            return
        f["t"] += dt
        k = ease(f["t"] / f["dur"])
        e = self.enemy
        target = self.player.left_anchor if f["to"] == "player" else e.to_world(e.skeleton()["handR"])
        pos = v_lerp(f["from"], target, k)
        f["pos"] = (pos[0], pos[1] + math.sin(math.pi * k) * 0.25, pos[2])
        a = k * math.pi * 4                              # меч кувыркается в полёте
        f["spin"] = v_norm((math.sin(a) * 0.9, math.cos(a), 0.3))
        if f["t"] >= f["dur"]:
            self.flying = None
            if f["to"] == "player":
                self.player.left_saber = Saber(ENEMY_COLOR, target, REST_L[1])
            elif e.state != "dead":
                e.regain_saber()
                e.set_state("retreat")

    def enemy_force_push(self):
        p = self.player
        self.fx.wave(self.enemy.to_world((-0.1, 1.45, 0.62)), v_add(EYE, (0, -0.2, -0.3)), (1, 0.5, 0.4))

        def land():
            if self.state != "fight":
                return
            p.hp -= FORCE_PUSH_DAMAGE
            p.knocked_until = time.time() + 0.8
            self.fx.shake, self.fx.damage = 1.0, 0.6
            self.say("Вас отбросило Силой!", (1, 0.5, 0.4), 1.2)
            if p.hp <= 0:
                self.lose()
        self.later(0.2, land)

    # --- финал ---
    def kill_enemy(self, disarmed):
        e = self.enemy
        e.hp = 0
        e.has_saber = False if disarmed else e.has_saber
        e.set_state("dead")
        self.fx.burst(e.to_world((0, 0.98, 0.1)), 60, (1, 0.6, 0.2), 2.5)
        self.state = "won"
        self.slowmo(1.2)
        self.say("Противник разрублен!" if disarmed else "Победа!", (1, 0.85, 0.3), 99)
        return "kill"

    def lose(self):
        self.state = "lost"
        self.enemy.idle_time = 1e9
        self.enemy.set_state("victory")
        self.say("Поражение", (1, 0.3, 0.3), 99)


# ======================= графика =======================

def make_glow_surface():
    surf = pygame.Surface((64, 4), pygame.SRCALPHA)
    for x in range(64):
        d = abs(x - 31.5) / 31.5
        a = int(255 * math.exp(-d * d * 6))
        for y in range(4):
            surf.set_at((x, y), (255, 255, 255, a))
    return surf


class Renderer:
    def __init__(self):
        self.q = gluNewQuadric()
        gluQuadricNormals(self.q, GLU_SMOOTH)
        self.glow = make_texture(make_glow_surface())
        self.arena = glGenLists(1)
        glNewList(self.arena, GL_COMPILE)
        self.build_arena()
        glEndList()
        self.eye = EYE

    # --- примитивы ---
    def cyl(self, a, b, r0, r1, slices=12):
        d = v_sub(b, a)
        L = v_len(d)
        if L < 1e-6:
            return
        glPushMatrix()
        glTranslatef(*a)
        axis = (-d[1], d[0], 0.0)
        if abs(axis[0]) + abs(axis[1]) < 1e-9:
            axis = (1.0, 0.0, 0.0)
        glRotatef(math.degrees(math.acos(clamp(d[2] / L, -1, 1))), *axis)
        gluCylinder(self.q, r0, r1, L, slices, 1)
        glPopMatrix()

    def sphere(self, p, r, slices=12):
        glPushMatrix()
        glTranslatef(*p)
        gluSphere(self.q, r, slices, max(6, slices * 2 // 3))
        glPopMatrix()

    def billboard(self, a, b, width, color, alpha):
        axis = v_norm(v_sub(b, a))
        a, b = v_sub(a, v_mul(axis, width * 0.4)), v_add(b, v_mul(axis, width * 0.4))
        view = v_norm(v_sub(v_lerp(a, b, 0.5), self.eye))
        side = v_cross(axis, view)
        if v_len(side) < 1e-4:
            side = v_cross(axis, (0, 1, 0))
        side = v_mul(v_norm(side), width / 2)
        glColor4f(color[0], color[1], color[2], alpha)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0.5); glVertex3f(*v_sub(a, side))  # noqa: E702
        glTexCoord2f(1, 0.5); glVertex3f(*v_add(a, side))  # noqa: E702
        glTexCoord2f(1, 0.5); glVertex3f(*v_add(b, side))  # noqa: E702
        glTexCoord2f(0, 0.5); glVertex3f(*v_sub(b, side))  # noqa: E702
        glEnd()

    # --- сцена ---
    def build_arena(self):
        glColor3f(0.13, 0.14, 0.17)            # пол, разбитый на сетку - его освещают мечи
        glNormal3f(0, 1, 0)
        glBegin(GL_QUADS)
        step = 0.25
        for i in range(-16, 16):
            for j in range(-26, 6):
                x, z = i * step, j * step
                for dx, dz in ((0, 0), (0, step), (step, step), (step, 0)):
                    glVertex3f(x + dx, 0, z + dz)
        glEnd()
        glDisable(GL_LIGHTING)
        glColor3f(0.04, 0.045, 0.06)
        glBegin(GL_QUADS)
        for x0, z0, x1, z1 in ((-20, -40, 20, -6.5), (-20, -6.5, -4, 1.5), (4, -6.5, 20, 1.5), (-20, 1.5, 20, 10)):
            glVertex3f(x0, -0.001, z0)
            glVertex3f(x0, -0.001, z1)
            glVertex3f(x1, -0.001, z1)
            glVertex3f(x1, -0.001, z0)
        glEnd()
        glColor3f(0.15, 0.35, 0.5)             # светящиеся круги на полу
        for r in (1.0, 2.5, 4.0, 5.5):
            glBegin(GL_LINE_LOOP)
            for i in range(96):
                a = 2 * math.pi * i / 96
                glVertex3f(r * math.cos(a), 0.003, -2.0 + r * math.sin(a))
            glEnd()
        for i in range(10):                    # колонны со светящимися полосами
            a = 2 * math.pi * i / 10 + 0.3
            cx, cz = 7.5 * math.cos(a), -2.0 + 7.5 * math.sin(a)
            if cz > 1.0:
                continue
            glEnable(GL_LIGHTING)
            glColor3f(0.1, 0.1, 0.12)
            glPushMatrix()
            glTranslatef(cx, 0, cz)
            glRotatef(-90, 1, 0, 0)
            gluCylinder(self.q, 0.45, 0.45, 7, 12, 1)
            glPopMatrix()
            glDisable(GL_LIGHTING)
            glColor3f(0.9, 0.35, 0.1)
            nx, nz = -math.cos(a), -math.sin(a)
            px, pz = cx + nx * 0.46, cz + nz * 0.46
            tx, tz = -nz * 0.06, nx * 0.06
            glBegin(GL_QUADS)
            glVertex3f(px - tx, 0.3, pz - tz)
            glVertex3f(px + tx, 0.3, pz + tz)
            glVertex3f(px + tx, 6.5, pz + tz)
            glVertex3f(px - tx, 6.5, pz - tz)
            glEnd()
        glEnable(GL_LIGHTING)

    def draw_enemy(self, e):
        sk = e.skeleton()
        glPushMatrix()
        glTranslatef(e.x, e.lift, e.z)
        glRotatef(math.degrees(e.yaw()), 0, 1, 0)
        glScalef(-1, 1, 1)                     # его правая рука - слева на экране
        dead = e.state == "dead"
        robe, skin = (0.09, 0.08, 0.09), (0.35, 0.3, 0.3)

        # нижняя часть: мантия и сапоги
        glPushMatrix()
        if dead:                               # ноги подкашиваются и падают вперёд
            k = ease((e.t - 0.5) / 0.9)
            glRotatef(80 * k, 1, 0, 0)
        stride = math.sin(e.walk) * 0.12
        dangle = math.sin(e.game.clock * 9) * 0.06 if e.choked else 0
        glColor3f(0.05, 0.05, 0.05)
        for side, off in ((1, stride + dangle), (-1, -stride - dangle)):
            glPushMatrix()
            glTranslatef(0.11 * side, 0.06, off)
            glScalef(0.09, 0.12, 0.22)
            self.sphere((0, 0, 0), 1, 10)
            glPopMatrix()
        glColor3f(*robe)
        glPushMatrix()
        glTranslatef(0, 0.1, 0)
        glRotatef(-90, 1, 0, 0)
        gluCylinder(self.q, 0.3, 0.17, 0.9, 16, 1)
        glPopMatrix()
        if dead:
            self.cut_glow((0, 0.98, 0), e.t)
        glPopMatrix()

        # верхняя часть: корпус, голова, руки
        glPushMatrix()
        if dead:                               # верх отлетает назад
            k = ease(e.t / 1.1)
            glTranslatef(0.05 * k, -0.7 * k, -0.35 * k)
            glTranslatef(0, 0.95, 0)
            glRotatef(-95 * k, 1, 0, 0)
            glTranslatef(0, -0.95, 0)
        glColor3f(*robe)
        self.cyl(sk["hip"], sk["neck"], 0.17, 0.2, 14)
        glColor3f(0.2, 0.08, 0.06)
        self.cyl(sk["lean"]((0, 1.0, 0)), sk["lean"]((0, 1.07, 0)), 0.172, 0.172, 14)   # пояс
        glColor3f(*robe)
        for sh in ("shR", "shL"):
            self.sphere(sk[sh], 0.075, 10)
        for side in "RL":
            self.cyl(sk["sh" + side], sk["elbow" + side], 0.055, 0.05, 10)
            self.cyl(sk["elbow" + side], sk["hand" + side], 0.05, 0.065, 10)
            glColor3f(*skin)
            self.sphere(sk["hand" + side], 0.04, 10)
            glColor3f(*robe)
        lean = sk["lean"]
        self.sphere(lean((0, 1.66, -0.015)), 0.128, 14)        # капюшон
        glColor3f(0.16, 0.13, 0.13)
        self.sphere(lean((0, 1.65, 0.03)), 0.1, 14)            # лицо
        glDisable(GL_LIGHTING)
        glColor3f(1, 0.8, 0.1)                                 # светящиеся глаза
        for x in (-0.035, 0.035):
            self.sphere(lean((x, 1.67, 0.122)), 0.012, 8)
        glEnable(GL_LIGHTING)
        if e.has_saber and not dead:
            hand, d = sk["handR"], sk["blade"]
            self.hilt(hand, d)
        if dead:
            self.cut_glow((0, 0.98, 0), e.t)
        glPopMatrix()
        glPopMatrix()

    def cut_glow(self, p, t):
        k = max(0.0, 1 - t / 2.5)
        if k <= 0:
            return
        glDisable(GL_LIGHTING)
        glColor3f(1.0 * k, 0.45 * k, 0.1 * k)
        glPushMatrix()
        glTranslatef(*p)
        glRotatef(-90, 1, 0, 0)
        gluDisk(self.q, 0, 0.17, 16, 1)
        glPopMatrix()
        glEnable(GL_LIGHTING)

    def hilt(self, hand, d):
        glColor3f(0.7, 0.7, 0.75)
        self.cyl(v_sub(hand, v_mul(d, HILT_BACK)), v_add(hand, v_mul(d, HILT_FRONT)), 0.016, 0.016, 10)
        glColor3f(0.08, 0.08, 0.08)
        self.cyl(v_sub(hand, v_mul(d, 0.06)), v_add(hand, v_mul(d, 0.06)), 0.018, 0.018, 10)
        glColor3f(0.85, 0.85, 0.9)
        self.cyl(v_add(hand, v_mul(d, HILT_FRONT - 0.03)), v_add(hand, v_mul(d, HILT_FRONT)), 0.02, 0.022, 10)

    def draw_hand(self, view, side, fist_at=None):
        skin, sleeve = (0.85, 0.65, 0.52), (0.42, 0.3, 0.18)
        if view is None:
            if fist_at is None:
                return
            wrist = v_add(fist_at, (0.02 * side, -0.05, 0.02))
            glColor3f(*skin)
            self.sphere(fist_at, 0.04, 12)
        else:
            pts = view[0]
            wrist = pts[0]
            glColor3f(*skin)
            for a, b in CONNECTIONS:
                self.cyl(pts[a], pts[b], 0.0085, 0.0085, 8)
            for p in pts:
                self.sphere(p, 0.0095, 8)
            n = v_norm(v_cross(v_sub(pts[5], pts[0]), v_sub(pts[17], pts[0])))
            if n[2] < 0:
                n = v_mul(n, -1)
            glNormal3f(*n)
            glBegin(GL_POLYGON)
            for i in (0, 1, 5, 9, 13, 17):
                glVertex3f(*pts[i])
            glEnd()
        glColor3f(*sleeve)
        self.cyl(wrist, v_add(wrist, (0.07 * side, -0.4, 0.3)), 0.032, 0.06, 12)

    def render(self, game, mode):
        p, e, fx = game.player, game.enemy, game.fx
        sh = fx.shake * 0.04
        self.eye = v_add(EYE, (random.uniform(-sh, sh), random.uniform(-sh, sh), 0))
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(FOV, ASPECT, 0.05, 60)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(*self.eye, self.eye[0], self.eye[1], self.eye[2] - 1, 0, 1, 0)

        glLightfv(GL_LIGHT0, GL_POSITION, (0.3, 1.0, 0.5, 0))
        lights = [(GL_LIGHT1, p.right, PLAYER_COLOR)]
        if p.left_saber:
            lights.append((GL_LIGHT3, p.left_saber, ENEMY_COLOR))
        else:
            glDisable(GL_LIGHT3)
        for light, s, c in lights:
            glEnable(light)
            glLightfv(light, GL_POSITION, (*v_lerp(s.base(), s.tip(), 0.5), 1))
            glLightfv(light, GL_DIFFUSE, (*c, 1))
        esw = e.saber_world()
        glow_src = (v_lerp(*esw, 0.5) if esw else game.flying["pos"] if game.flying else None)
        if glow_src:
            glEnable(GL_LIGHT2)
            glLightfv(GL_LIGHT2, GL_POSITION, (*glow_src, 1))
        else:
            glDisable(GL_LIGHT2)
        if fx.flash:
            glEnable(GL_LIGHT4)
            glLightfv(GL_LIGHT4, GL_POSITION, (*fx.flash[0], 1))
            k = fx.flash[1] * 2
            glLightfv(GL_LIGHT4, GL_DIFFUSE, (fx.flash[2][0] * k, fx.flash[2][1] * k, fx.flash[2][2] * k, 1))
        else:
            glDisable(GL_LIGHT4)

        glEnable(GL_FOG)
        glCallList(self.arena)
        self.draw_enemy(e)
        glDisable(GL_FOG)

        # руки игрока
        self.draw_hand(p.right_view, 1, None if p.right_view else p.right.hilt)
        self.hilt(p.right.hilt, p.right.dir)
        left_fist = p.left_saber.hilt if p.left_saber else (None if mode == "camera" else REST_L[0])
        self.draw_hand(p.left_view, -1, None if p.left_view else left_fist)
        if p.left_saber:
            self.hilt(p.left_saber.hilt, p.left_saber.dir)
        if game.flying:
            self.hilt(game.flying["pos"], game.flying.get("spin", game.flying["dir"]))

        self.draw_glow(game)

    def draw_glow(self, game):
        """Всё светящееся: клинки, следы, искры, волны Силы (аддитивное смешивание)."""
        p, e, fx = game.player, game.enemy, game.fx
        glDisable(GL_LIGHTING)
        glDepthMask(GL_FALSE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)

        blades = [(s.base(), s.tip(), s.color, s.trail) for s in p.sabers()]
        esw = e.saber_world()
        if esw:
            blades.append((esw[0], esw[1], ENEMY_COLOR, e.trail))
        f = game.flying
        if f:
            d = f.get("spin", f["dir"])
            blades.append((v_add(f["pos"], v_mul(d, HILT_FRONT)), v_add(f["pos"], v_mul(d, HILT_FRONT + BLADE_LEN)),
                           ENEMY_COLOR, ()))
        for base, tip, color, trail in blades:      # след взмаха
            tr = list(trail)
            if len(tr) > 1:
                glBegin(GL_QUAD_STRIP)
                for i, (b0, t0) in enumerate(tr):
                    glColor4f(*color, 0.22 * i / len(tr))
                    glVertex3f(*b0)
                    glVertex3f(*t0)
                glEnd()
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.glow)
        for base, tip, color, _ in blades:
            self.billboard(base, tip, 0.17, color, 0.35)
            self.billboard(base, tip, 0.065, color, 0.9)
            self.billboard(base, tip, 0.022, v_lerp(color, (1, 1, 1), 0.8), 1.0)
        if game.state == "fight" and e.state == "windup" and HINTS[0]:   # призрак будущего удара
            frames, impact = ATTACKS[e.attack]
            hand, d = frames[impact]
            hw, dw = e.to_world(hand), e.dir_world(d)
            pulse = 0.25 + 0.15 * math.sin(game.clock * 20)
            self.billboard(v_add(hw, v_mul(dw, HILT_FRONT)), v_add(hw, v_mul(dw, HILT_FRONT + BLADE_LEN)),
                           0.05, ENEMY_COLOR, pulse)
        glDisable(GL_TEXTURE_2D)

        glLineWidth(2)
        glBegin(GL_LINES)
        for pos, vel, life, color in fx.sparks:
            glColor4f(*color, min(1.0, life * 3))
            glVertex3f(*pos)
            glVertex3f(*v_sub(pos, v_mul(vel, 0.025)))
        glEnd()
        for w in fx.waves:                          # волны Силы - расширяющиеся кольца
            for j in range(3):
                k = clamp((w["t"] - j * 0.04) / w["dur"], 0, 1)
                if k <= 0:
                    continue
                c = v_lerp(w["a"], w["b"], k)
                r = 0.05 + 0.5 * k
                fade = (1 - k) * 0.6 + 0.2
                glColor4f(*w["color"], fade)
                glBegin(GL_LINE_LOOP)
                for i in range(32):
                    a = 2 * math.pi * i / 32
                    glVertex3f(c[0] + r * math.cos(a), c[1] + r * math.sin(a), c[2])
                glEnd()
        if e.choked:                                # аура удушения на шее
            neck = e.to_world((0, 1.52, 0.02))
            r = 0.13 + 0.02 * math.sin(game.clock * 12)
            glColor4f(0.5, 0.75, 1, 0.7)
            glBegin(GL_LINE_LOOP)
            for i in range(24):
                a = 2 * math.pi * i / 24
                glVertex3f(neck[0] + r * math.cos(a), neck[1], neck[2] + r * math.sin(a))
            glEnd()

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_TRUE)
        glEnable(GL_LIGHTING)


HINTS = [True]   # показывать призрак удара (клавиша H)


def draw_hud(hud, game, mode, tracker, gest, show_help):
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, W, 0, H, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    p, e, fx = game.player, game.enemy, game.fx
    now = time.time()

    if fx.damage > 0:
        hud.rect(0, 0, W, H, (0.8, 0, 0, fx.damage * 0.35))
    if game.time_scale < 1:
        hud.rect(0, 0, W, H, (0.1, 0.25, 0.5, 0.15))

    hud.text("Тёмный воин", W / 2, 14, 22, (1, 0.6, 0.55), center=True)
    hud.rect(W / 2 - 250, 44, 500, 14, (0, 0, 0, 0.6))
    hud.rect(W / 2 - 250, 44, 500 * max(0, e.hp) / ENEMY_HP, 14, (0.85, 0.15, 0.1, 0.95))

    hud.rect(16, H - 92, 300, 76, (0, 0, 0, 0.45))
    hud.text("Здоровье", 26, H - 88, 18)
    hud.rect(120, H - 82, 186, 12, (0.2, 0.2, 0.2, 0.8))
    hud.rect(120, H - 82, 186 * max(0, p.hp) / PLAYER_HP, 12, (0.3, 0.85, 0.35, 0.95))
    hud.text("Сила", 26, H - 62, 18)
    hud.rect(120, H - 56, 186, 12, (0.2, 0.2, 0.2, 0.8))
    hud.rect(120, H - 56, 186 * p.force / 100, 12, (0.35, 0.6, 1, 0.95))
    left = ("в руке трофейный меч" if p.left_saber else
            {"open": "ладонь", "fist": "кулак", "pinch": "щипок: удушение", "other": "..."}.get(gest.pose, "не видно"))
    hud.text(f"Левая: {left}", 26, H - 38, 17, (0.8, 0.85, 1))
    if gest.charge > 0 and not p.left_saber:     # заряд вырывания меча
        k = gest.charge / DISARM_CHARGE
        hud.rect(26, H - 112, 200 * k, 8, (1, 0.85, 0.3, 0.9) if k >= 1 else (0.7, 0.7, 1, 0.8))
        hud.text("сожмите кулак!" if k >= 1 else "захват меча...", 232, H - 120, 16, (1, 0.9, 0.5))

    text, color, until = game.message
    if now < until:
        hud.text(text, W / 2, 90, 46 if game.state != "fight" else 36, color, center=True)
    if e.state == "windup" and game.state == "fight":
        hud.text(f"{e.attack.capitalize()}!", W / 2, 140, 24, (1, 0.5, 0.4), center=True)

    if mode == "camera":
        if tracker.preview:
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glWindowPos2d(W - PREVIEW_W - 12, 12)
            glDrawPixels(PREVIEW_W, PREVIEW_H, GL_RGB, GL_UNSIGNED_BYTE, tracker.preview)
        if not tracker.ready:
            hud.text("Запуск камеры...", W / 2, H / 2, 36, center=True)
    if show_help:
        lines = ["ПРАВАЯ - меч: наклон кисти = наклон клинка",
                 "  блок - клинок поперёк красного призрака удара",
                 "  резкий блок = парирование,  резкий взмах по врагу = удар",
                 "ЛЕВАЯ - Сила:",
                 "  ладонь резко вперёд - толчок,  резко к себе - притяжение",
                 "  щипок (OK) держать - удушение (замедляет)",
                 "  ладонь неподвижно, затем кулак - вырвать меч",
                 "M мышь/камера  X поменять руки  H призрак  Tab справка  R заново"]
        hud.rect(W - 560, 70, 548, 26 * len(lines) + 16, (0, 0, 0, 0.5))
        for i, line in enumerate(lines):
            hud.text(line, W - 548, 78 + i * 26, 18, (0.9, 0.9, 0.9))
    if game.state != "fight":
        hud.text("R - новый бой", W / 2, 150, 26, center=True)

    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)


def init_gl():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_LIGHTING)
    glEnable(GL_NORMALIZE)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.12, 0.12, 0.15, 1))
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.3, 0.32, 0.42, 1))
    for light, color in ((GL_LIGHT1, PLAYER_COLOR), (GL_LIGHT2, ENEMY_COLOR), (GL_LIGHT3, ENEMY_COLOR),
                         (GL_LIGHT4, (1, 1, 1))):
        glLightfv(light, GL_DIFFUSE, (*color, 1))
        glLightf(light, GL_CONSTANT_ATTENUATION, 0.3)
        glLightf(light, GL_LINEAR_ATTENUATION, 0.6)
        glLightf(light, GL_QUADRATIC_ATTENUATION, 1.2)
    glFogi(GL_FOG_MODE, GL_EXP2)
    glFogf(GL_FOG_DENSITY, 0.09)
    glFogfv(GL_FOG_COLOR, (0.02, 0.02, 0.04, 1))
    glClearColor(0.02, 0.02, 0.04, 1)


def main():
    pygame.init()
    create_window()
    pygame.display.set_caption("Дуэль джедаев - управление руками")
    init_gl()
    renderer, hud = Renderer(), Hud()
    tracker = TwoHandTracker()
    tracker.start()
    game = Game()
    right_f, left_f, gest = HandFilter(1), HandFilter(-1), ForceGestures()
    mode = "camera"
    mouse_angle = 0.0
    show_help, help_until = True, time.time() + 12
    clock = pygame.time.Clock()

    running = True
    while running:
        dt = min(clock.tick(60) / 1000, 0.05)
        events = []
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_m:
                    mode = "mouse" if mode == "camera" or tracker.error else "camera"
                elif ev.key == pygame.K_x:
                    tracker.swap = not tracker.swap
                    game.say("Руки поменяны местами", (0.8, 0.8, 0.8), 1)
                elif ev.key == pygame.K_TAB:
                    show_help, help_until = not show_help, 0
                elif ev.key == pygame.K_h:
                    HINTS[0] = not HINTS[0]
                    game.say("Призрак удара " + ("включён" if HINTS[0] else "выключен"), (0.8, 0.8, 0.8), 1)
                elif ev.key == pygame.K_r:
                    game.reset()
                elif ev.key in (pygame.K_1, pygame.K_2, pygame.K_4):
                    events.append({pygame.K_1: "push", pygame.K_2: "pull", pygame.K_4: "disarm"}[ev.key])
        if show_help and help_until and time.time() > help_until:
            show_help, help_until = False, 0

        if mode == "camera" and tracker.error:
            mode = "mouse"
            game.say(f"Камера недоступна ({tracker.error}) - играем мышью", (1, 0.5, 0.4), 4)
        keys = pygame.key.get_pressed()
        hands = {"events": events, "choke": keys[pygame.K_3], "right": None, "left": None, "right_pose": None}
        if mode == "camera":
            try:
                while True:
                    t, lm_l, lm_r = tracker.samples.get_nowait()
                    right_f.feed(t, lm_r)
                    left_f.feed(t, lm_l)
                    ev = gest.feed(t, lm_l)
                    if ev:
                        events.append(ev)
            except queue.Empty:
                pass
            hands["right"], hands["left"] = right_f.world(), left_f.world()
            hands["choke"] = hands["choke"] or gest.pose == "pinch"
        else:
            mouse_angle += (keys[pygame.K_d] - keys[pygame.K_a]) * dt * 2.5
            mx, my = pygame.mouse.get_pos()
            hilt = (EYE[0] + (mx / W - 0.5) * 1.0, EYE[1] - 0.1 - (my / H - 0.5) * 0.6, -0.55)
            hands["right_pose"] = (hilt, v_norm((math.sin(mouse_angle) * 0.85, math.cos(mouse_angle) * 0.85, -0.5)))

        game.update(dt, hands)
        renderer.render(game, mode)
        draw_hud(hud, game, mode, tracker, gest, show_help)
        pygame.display.flip()

    tracker.running = False
    pygame.quit()


if __name__ == "__main__":
    main()
