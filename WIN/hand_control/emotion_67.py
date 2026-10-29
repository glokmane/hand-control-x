# Copyright (c) 2026 Hand Control X authors. All rights reserved.
# Proprietary and confidential — see LICENSE. Do not redistribute.
"""
Эмоции по лицу + мем "67" руками через веб-камеру.

Лицо: MediaPipe Face Landmarker выдаёт 52 параметра мимики (улыбка, открытый рот, брови...),
по ним выбирается смайлик: смеюсь, радость, грусть, удивление, злость, подмигиваю, поцелуй, сплю.
Первые 2 секунды - калибровка: смотрите в камеру с обычным (нейтральным) лицом.

Руки: быстро двигайте обеими руками вверх-вниз попеременно (одна вверх, другая вниз) -
под смайликом появятся руки, делающие "67".

Клавиши: C - перекалибровать нейтральное лицо, Esc / Q - выход.
"""
import math
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import pygame
from mediapipe.tasks.python import BaseOptions, vision

from hand_control import CONNECTIONS, MIDDLE_MCP, MODEL_PATH, ensure_model

FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/latest/face_landmarker.task")
FACE_MODEL_PATH = Path(__file__).with_name("face_landmarker.task")

W, H = 1280, 720
CAM_RECT = pygame.Rect(16, 16, 800, 600)
CALIB_SECS = 2.0
SWITCH_DELAY = 0.2       # эмоция должна продержаться столько секунд, чтобы смайлик сменился

# --- мем 67 ---
WINDOW_67 = 1.2          # окно анализа, с
AMP_67 = 0.10            # минимальный размах "одна рука выше другой" (доля высоты кадра)
TURNS_67 = 3             # сколько смен направления должно быть в окне
HOLD_67 = 0.8            # сколько секунд держать картинку после последнего движения

EMOTIONS = {
    "laugh": "Смеюсь", "happy": "Радость", "sad": "Грусть", "surprise": "Удивление",
    "angry": "Злость", "wink": "Подмигиваю", "kiss": "Поцелуй", "sleepy": "Сплю", "neutral": "Нейтрально",
}


def ensure_face_model():
    if not FACE_MODEL_PATH.exists():
        print("Скачиваю модель MediaPipe face_landmarker...")
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)


# ======================= эмоции =======================

class EmotionDetector:
    """Параметры мимики -> эмоция, с калибровкой под нейтральное лицо и защитой от мерцания."""

    def __init__(self):
        self.recalibrate()
        self.shown = "neutral"
        self.changed_at = time.time()
        self.cand, self.cand_since = "neutral", 0.0
        self.closed_since = None
        self.signals = {}

    def recalibrate(self):
        self.base = None
        self.calib = []
        self.calib_start = None
        self.smooth = {}

    @property
    def calibrating(self):
        return self.base is None

    def update(self, shapes, t):
        if shapes is None:
            return self.shown
        if self.base is None:                       # набираем нейтральное лицо
            self.calib_start = self.calib_start or t
            self.calib.append(shapes)
            if t - self.calib_start >= CALIB_SECS:
                self.base = {k: sum(s[k] for s in self.calib) / len(self.calib) for k in shapes}
            return self.shown
        for k, v in shapes.items():                 # отклонение от своего нейтрального лица + сглаживание
            d = max(0.0, v - self.base.get(k, 0.0))
            self.smooth[k] = self.smooth.get(k, d) * 0.5 + d * 0.5
        emo = self.classify(self.smooth, t)
        if emo != self.cand:
            self.cand, self.cand_since = emo, t
        elif emo != self.shown and t - self.cand_since >= SWITCH_DELAY:
            self.shown, self.changed_at = emo, time.time()
        return self.shown

    def classify(self, s, t):
        g = lambda *keys: sum(s.get(k, 0.0) for k in keys) / len(keys)  # noqa: E731
        smile = g("mouthSmileLeft", "mouthSmileRight")
        jaw = g("jawOpen")
        frown = g("mouthFrownLeft", "mouthFrownRight")
        brow_up = max(g("browInnerUp"), g("browOuterUpLeft", "browOuterUpRight"))
        brow_down = g("browDownLeft", "browDownRight")
        wide = g("eyeWideLeft", "eyeWideRight")
        blink_l, blink_r = g("eyeBlinkLeft"), g("eyeBlinkRight")
        pucker = g("mouthPucker")
        sneer = g("noseSneerLeft", "noseSneerRight")
        self.signals = {"улыбка": smile, "рот открыт": jaw, "уголки вниз": frown, "брови вверх": brow_up,
                        "брови нахмурены": brow_down, "глаза широко": wide,
                        "глаза закрыты": min(blink_l, blink_r), "губы трубочкой": pucker}

        both_closed = blink_l > 0.5 and blink_r > 0.5
        self.closed_since = (self.closed_since or t) if both_closed else None

        if smile > 0.45 and jaw > 0.2:
            return "laugh"
        if jaw > 0.35 and smile < 0.25 and (wide > 0.15 or brow_up > 0.25):
            return "surprise"
        if abs(blink_l - blink_r) > 0.4 and max(blink_l, blink_r) > 0.5:
            return "wink"
        if self.closed_since and t - self.closed_since > 0.8:
            return "sleepy"
        if pucker > 0.45 and jaw < 0.3:
            return "kiss"
        if smile > 0.35:
            return "happy"
        if (brow_down > 0.25 or sneer > 0.35) and smile < 0.2:
            return "angry"
        if frown > 0.15 or (g("browInnerUp") > 0.3 and jaw < 0.15):
            return "sad"
        return "neutral"


# ======================= мем 67 =======================

class SixSeven:
    """Ловит попеременное движение рук вверх-вниз (одна поднимается, другая опускается)."""

    def __init__(self):
        self.hist = deque()          # (t, y левой, y правой)
        self.turns = deque()         # моменты смены направления
        self.dir, self.ext = 0, None
        self.last_active = -1e9
        self.y = None                # последние (yL, yR) - для анимации рук

    def update(self, t, ys):
        if len(ys) != 2:
            self.hist.clear()
            self.dir, self.ext = 0, None
            self.y = None
            return self.active(t)
        yl, yr = ys
        self.y = (yl, yr)
        self.hist.append((t, yl, yr))
        while self.hist and t - self.hist[0][0] > WINDOW_67:
            self.hist.popleft()
        while self.turns and t - self.turns[0] > WINDOW_67:
            self.turns.popleft()

        # смены направления разности высот рук, с гистерезисом против дрожания
        d = yl - yr
        gap = AMP_67 * 0.4
        if self.ext is None:
            self.ext = d
        elif self.dir == 0:                       # ещё не знаем направление
            if abs(d - self.ext) > gap:
                self.dir, self.ext = (1 if d > self.ext else -1), d
        elif (d - self.ext) * self.dir > 0:       # продолжаем в ту же сторону
            self.ext = d
        elif abs(d - self.ext) > gap:             # развернулись
            self.turns.append(t)
            self.dir, self.ext = -self.dir, d

        ds = [h[1] - h[2] for h in self.hist]
        amp = max(ds) - min(ds) if ds else 0
        if len(self.turns) >= TURNS_67 and amp >= AMP_67 and self.antiphase():
            self.last_active = t
        return self.active(t)

    def antiphase(self):
        """Руки движутся в противоположные стороны: скорости отрицательно скоррелированы."""
        h = list(self.hist)
        vl = [b[1] - a[1] for a, b in zip(h, h[1:])]
        vr = [b[2] - a[2] for a, b in zip(h, h[1:])]
        den = math.sqrt(sum(v * v for v in vl) * sum(v * v for v in vr))
        return den > 1e-9 and sum(a * b for a, b in zip(vl, vr)) / den < -0.3

    def active(self, t):
        return t - self.last_active < HOLD_67


# ======================= рисование смайликов =======================

N = 600                       # смайлик рисуется крупно и плавно уменьшается - так края гладкие
C = (300, 300)
R = 255
YELLOW, EDGE, DARK = (255, 204, 77), (230, 155, 35), (92, 56, 28)


def thick_arc(s, color, center, rx, ry, a0, a1, w):
    """Толстая дуга с круглыми концами (углы как в математике: 0 - вправо, pi/2 - вверх)."""
    pts = [(center[0] + rx * math.cos(a0 + (a1 - a0) * i / 40),
            center[1] - ry * math.sin(a0 + (a1 - a0) * i / 40)) for i in range(41)]
    pygame.draw.lines(s, color, False, pts, w)
    for p in pts:
        pygame.draw.circle(s, color, p, w / 2)


def thick_line(s, color, a, b, w):
    pygame.draw.line(s, color, a, b, w)
    pygame.draw.circle(s, color, a, w / 2)
    pygame.draw.circle(s, color, b, w / 2)


def eye(s, x, y, w=50, h=74):
    pygame.draw.ellipse(s, DARK, (x - w / 2, y - h / 2, w, h))
    pygame.draw.circle(s, (255, 255, 255), (x - w * 0.18, y - h * 0.22), w * 0.17)


def tear(s, x, y, r=20):
    color = (90, 175, 255)
    pygame.draw.polygon(s, color, [(x, y - r * 2.2), (x - r * 0.9, y - r * 0.3), (x + r * 0.9, y - r * 0.3)])
    pygame.draw.circle(s, color, (x, y), r)
    pygame.draw.circle(s, (200, 235, 255), (x - r * 0.35, y - r * 0.3), r * 0.3)


def heart(s, x, y, r=24, color=(240, 60, 90)):
    pygame.draw.circle(s, color, (x - r * 0.55, y), r * 0.65)
    pygame.draw.circle(s, color, (x + r * 0.55, y), r * 0.65)
    pygame.draw.polygon(s, color, [(x - r * 1.18, y + r * 0.2), (x + r * 1.18, y + r * 0.2), (x, y + r * 1.5)])


def blush(s):
    layer = pygame.Surface((N, N), pygame.SRCALPHA)
    for x in (165, 435):
        pygame.draw.ellipse(layer, (255, 120, 120, 90), (x - 50, 300, 100, 55))
    s.blit(layer, (0, 0))


def draw_emoji(emotion, t, font_z):
    s = pygame.Surface((N, N), pygame.SRCALPHA)
    face = (250, 120, 60) if emotion == "angry" else YELLOW
    edge = (205, 70, 40) if emotion == "angry" else EDGE
    pygame.draw.circle(s, edge, C, R + 10)
    pygame.draw.circle(s, face, C, R)
    shine = pygame.Surface((N, N), pygame.SRCALPHA)
    pygame.draw.ellipse(shine, (255, 255, 255, 45), (135, 75, 200, 110))
    s.blit(shine, (0, 0))

    if emotion == "neutral":
        eye(s, 205, 245)
        eye(s, 395, 245)
        thick_line(s, DARK, (215, 405), (385, 405), 24)
    elif emotion == "happy":
        thick_arc(s, DARK, (205, 260), 45, 38, 0.25, math.pi - 0.25, 20)
        thick_arc(s, DARK, (395, 260), 45, 38, 0.25, math.pi - 0.25, 20)
        blush(s)
        thick_arc(s, DARK, (300, 330), 125, 100, math.pi + 0.35, 2 * math.pi - 0.35, 22)
    elif emotion == "laugh":
        pygame.draw.ellipse(s, (110, 30, 40), (165, 270, 270, 200))          # рот "D"
        pygame.draw.rect(s, face, (150, 260, 300, 105))
        pygame.draw.rect(s, (255, 255, 255), (190, 365, 220, 30), border_radius=8)
        pygame.draw.ellipse(s, (240, 110, 120), (245, 405, 110, 52))
        for sx in (1, -1):                                                      # глаза > <
            x0 = 300 - sx * 130
            pts = [(x0, 205), (x0 + sx * 55, 240), (x0, 275)]
            thick_line(s, DARK, pts[0], pts[1], 20)
            thick_line(s, DARK, pts[1], pts[2], 20)
        bob = math.sin(t * 9) * 10
        tear(s, 125, 300 + bob, 24)
        tear(s, 475, 300 - bob, 24)
    elif emotion == "sad":
        thick_line(s, DARK, (165, 180), (255, 150), 16)                       # брови домиком
        thick_line(s, DARK, (435, 180), (345, 150), 16)
        eye(s, 205, 260, 46, 64)
        eye(s, 395, 260, 46, 64)
        thick_arc(s, DARK, (300, 455), 95, 70, 0.35, math.pi - 0.35, 22)
        k = (t * 0.9) % 1.0
        tear(s, 190, 305 + k * 150, 20)
    elif emotion == "surprise":
        thick_arc(s, DARK, (205, 160), 52, 26, 0.3, math.pi - 0.3, 16)
        thick_arc(s, DARK, (395, 160), 52, 26, 0.3, math.pi - 0.3, 16)
        eye(s, 205, 245, 62, 86)
        eye(s, 395, 245, 62, 86)
        pygame.draw.ellipse(s, (110, 30, 40), (250, 355, 100, 120))
    elif emotion == "angry":
        thick_line(s, DARK, (160, 170), (265, 215), 26)                       # брови галочкой
        thick_line(s, DARK, (440, 170), (335, 215), 26)
        eye(s, 210, 270, 44, 56)
        eye(s, 390, 270, 44, 56)
        thick_arc(s, DARK, (300, 445), 85, 45, 0.3, math.pi - 0.3, 22)
    elif emotion == "wink":
        eye(s, 205, 245)
        thick_arc(s, DARK, (395, 255), 45, 30, 0.25, math.pi - 0.25, 20)
        thick_arc(s, DARK, (300, 340), 110, 85, math.pi + 0.4, 2 * math.pi - 0.4, 22)
        pygame.draw.ellipse(s, (240, 110, 120), (325, 395, 60, 50))           # высунутый язык
    elif emotion == "kiss":
        thick_arc(s, DARK, (205, 255), 45, 35, 0.25, math.pi - 0.25, 18)
        eye(s, 395, 245)
        thick_arc(s, DARK, (300, 375), 30, 24, -math.pi / 2 - 0.3, math.pi / 2 + 0.3, 18)   # губы "3"
        thick_arc(s, DARK, (300, 421), 30, 24, -math.pi / 2 - 0.3, math.pi / 2 + 0.3, 18)
        blush(s)
        heart(s, 450, 385 - (t * 60) % 80, 30)
    elif emotion == "sleepy":
        thick_arc(s, DARK, (205, 255), 45, 22, math.pi + 0.2, 2 * math.pi - 0.2, 18)
        thick_arc(s, DARK, (395, 255), 45, 22, math.pi + 0.2, 2 * math.pi - 0.2, 18)
        pygame.draw.ellipse(s, (110, 30, 40), (275, 380, 50, 60))
        for i, size in enumerate((1.0, 0.8, 0.6)):                            # Z z z
            k = (t * 0.5 + i / 3) % 1.0
            z = font_z.render("Z", True, (90, 120, 220))
            z = pygame.transform.rotozoom(z, 15, size * (0.6 + k * 0.6))
            z.set_alpha(int(255 * (1 - k)))
            s.blit(z, z.get_rect(center=(430 + i * 30 + k * 40, 170 - k * 120)))
    return s


def draw_hand(s, cx, cy, side):
    """Мультяшная ладонь пальцами вверх; side = -1 левая, +1 правая (большой палец внутрь)."""
    shapes = [pygame.Rect(cx - 52, cy - 15, 104, 95)]
    for i, (dx, h) in enumerate(((-39, 70), (-13, 84), (13, 80), (39, 66))):
        shapes.append(pygame.Rect(cx + dx - 12, cy - h + 5, 24, h))
    thumb = pygame.Rect(0, 0, 70, 30)
    thumb.center = (cx + side * 62, cy + 30)
    for r in shapes + [thumb]:                 # обводка
        pygame.draw.rect(s, EDGE, r.inflate(10, 10), border_radius=18)
    for r in shapes + [thumb]:
        pygame.draw.rect(s, YELLOW, r, border_radius=14)


def draw_67(t, ys, font_big, font_small):
    s = pygame.Surface((420, 330), pygame.SRCALPHA)
    if ys:                                     # руки на картинке повторяют настоящие
        mid = (ys[0] + ys[1]) / 2
        off = [max(-45, min(45, (y - mid) * 400)) for y in ys]
    else:
        off = [math.sin(t * 12) * 35, -math.sin(t * 12) * 35]
    for i, (cx, side) in enumerate(((110, 1), (310, -1))):
        cy = 190 + off[i]
        for k in (0, 1):                       # штрихи движения под ладонью
            y = cy + 98 + k * 10
            pygame.draw.line(s, (255, 255, 255), (cx - 30 + k * 10, y), (cx + 30 - k * 10, y), 3)
        draw_hand(s, cx, cy, side)
        num = font_big.render("67"[i], True, (255, 255, 255))
        outline = font_big.render("67"[i], True, (230, 60, 90))
        r = num.get_rect(center=(cx, cy - 112))
        for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3)):
            s.blit(outline, r.move(dx, dy))
        s.blit(num, r)
    return s


# ======================= приложение =======================

def main():
    ensure_model()
    ensure_face_model()
    face_lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=FACE_MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True))
    hand_lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5))
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit("Не удалось открыть веб-камеру")

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Эмоции и 67")
    font = lambda size: pygame.font.SysFont("segoeui,arial", size, bold=True)  # noqa: E731
    f_label, f_text, f_small, f_big, f_z = font(40), font(22), font(16), font(64), font(90)

    emo, six = EmotionDetector(), SixSeven()
    t0, last_ts = time.time(), -1
    fade_67 = 0.0
    clock = pygame.time.Clock()
    running = True
    while running:
        dt = clock.tick(60) / 1000
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q)):
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_c:
                emo.recalibrate()
        ok, frame = cap.read()
        if not ok:
            continue
        t = time.time()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = max(int((t - t0) * 1000), last_ts + 1)
        last_ts = ts
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        fr = face_lm.detect_for_video(image, ts)
        hr = hand_lm.detect_for_video(image, ts)

        shapes = ({c.category_name: c.score for c in fr.face_blendshapes[0]} if fr.face_blendshapes else None)
        emotion = emo.update(shapes, t)
        hands = sorted(hr.hand_landmarks, key=lambda lm: lm[MIDDLE_MCP].x)
        active_67 = six.update(t, [lm[MIDDLE_MCP].y for lm in hands])
        fade_67 = min(1.0, fade_67 + dt * 5) if active_67 else max(0.0, fade_67 - dt * 3)

        # камера с точками лица и скелетом рук
        h, w = frame.shape[:2]
        if fr.face_landmarks:
            for p in fr.face_landmarks[0][::2]:
                cv2.circle(rgb, (int(p.x * w), int(p.y * h)), 1, (120, 220, 255), -1)
        for lm in hands:
            pts = [(int(p.x * w), int(p.y * h)) for p in lm]
            for a, b in CONNECTIONS:
                cv2.line(rgb, pts[a], pts[b], (80, 255, 120), 2)
            for p in pts:
                cv2.circle(rgb, p, 3, (255, 90, 90), -1)
        cam = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")

        screen.fill((24, 24, 32))
        screen.blit(pygame.transform.smoothscale(cam, CAM_RECT.size), CAM_RECT)
        if shapes is None:
            msg = f_label.render("Лицо не найдено", True, (255, 120, 100))
            screen.blit(msg, msg.get_rect(center=(CAM_RECT.centerx, CAM_RECT.top + 40)))
        elif emo.calibrating:
            left = max(0.0, CALIB_SECS - (t - (emo.calib_start or t)))
            msg = f_label.render(f"Калибровка: смотрите спокойно... {left:.1f}", True, (255, 230, 120))
            screen.blit(msg, msg.get_rect(center=(CAM_RECT.centerx, CAM_RECT.top + 40)))

        for i, (name, v) in enumerate(emo.signals.items()):          # шкалы мимики
            x, y = 16 + (i % 4) * 202, 632 + (i // 4) * 40
            screen.blit(f_small.render(name, True, (190, 190, 200)), (x, y))
            pygame.draw.rect(screen, (55, 55, 70), (x, y + 22, 185, 10), border_radius=5)
            pygame.draw.rect(screen, (110, 200, 255), (x, y + 22, int(185 * min(1.0, v)), 10), border_radius=5)

        # смайлик: появляется с "пружинкой", при смехе трясётся
        since = time.time() - emo.changed_at
        scale = 1 - 0.35 * math.exp(-since * 7) * math.cos(since * 16)
        size = int(330 * scale)
        img = pygame.transform.smoothscale(draw_emoji(emotion, t, f_z), (size, size))
        shake = math.sin(t * 40) * 6 if emotion == "laugh" else 0
        cx, cy = 1052, 195 - int(fade_67 * 45)
        screen.blit(img, img.get_rect(center=(cx + shake, cy)))
        label = f_label.render(EMOTIONS[emotion], True, (255, 255, 255))
        screen.blit(label, label.get_rect(center=(cx, cy + 190)))

        if fade_67 > 0:
            pic = draw_67(t, six.y, f_big, f_text)
            pic.set_alpha(int(255 * fade_67))
            screen.blit(pic, pic.get_rect(center=(cx, cy + 375)))
        else:
            for i, line in enumerate(("Быстро двигайте руками вверх-вниз", "попеременно - будет 67")):
                txt = f_text.render(line, True, (130, 130, 150))
                screen.blit(txt, txt.get_rect(center=(cx, 560 + i * 30)))
        hint = f_small.render("C - перекалибровать лицо    Esc - выход", True, (120, 120, 140))
        screen.blit(hint, (840, H - 30))
        pygame.display.flip()

    cap.release()
    face_lm.close()
    hand_lm.close()
    pygame.quit()


if __name__ == "__main__":
    main()
