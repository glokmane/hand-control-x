# Hand Control X

**Control your computer with hand gestures and voice — using nothing but a webcam.**

Hand Control X turns an ordinary camera into a mouse, a remote and a game controller.
Move the cursor with your palm, click by pinching, scroll with a "V" sign, launch apps
or change the volume by talking, and play hand-driven mini-games on top of it.

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Windows" src="https://img.shields.io/badge/platform-Windows-0078D4?logo=windows&logoColor=white">
  <img alt="macOS planned" src="https://img.shields.io/badge/platform-macOS_planned-lightgrey?logo=apple&logoColor=white">
  <img alt="Arch Linux planned" src="https://img.shields.io/badge/platform-Arch_Linux_planned-lightgrey?logo=archlinux&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-proprietary-red">
</p>

> **Status:** Windows is the working platform. macOS and Arch Linux ports are in progress
> (see [`MAC/`](MAC) and [`ARCHLINUX/`](ARCHLINUX)).

[🇷🇺 Русская версия](README.ru.md)

<!--
  Add a demo GIF here — it is the single most important part of this README.
  Record 10–15 seconds (ScreenToGif / OBS), keep it under ~5 MB, then:
  ![Demo](docs/demo.gif)
-->

## Features

- **Cursor control** — move your hand, the cursor moves like a real mouse with a given DPI
  (1 inch of hand movement = DPI pixels).
- **Clicks, drag & drop, scrolling** — pinch gestures, no extra hardware.
- **Single-hand ownership** — the program locks onto the hand that started controlling it,
  so hands of other people in frame are ignored.
- **Offline Russian voice control** — [Vosk](https://alphacephei.com/vosk/) runs locally,
  no cloud, no account.
- **Browser control by voice** — search, tabs, links, scrolling, dictation, fullscreen.
- **Windows control by voice** — launch apps, window management, virtual desktops,
  volume, brightness, battery status, screenshots, lock/sleep/shutdown with confirmation.
- **Yandex Music integration** — next/previous track, seek, "what's playing", even when
  the tab is in the background.
- **Hand-driven games** — 3D darts, a first-person Jedi duel and an emotion-driven meme demo.
- **Auto-downloading models** — MediaPipe and Vosk models are fetched on first launch.

## Requirements

| | |
|---|---|
| OS | Windows 10/11 (primary), macOS and Arch Linux — planned |
| Python | 3.10 – 3.14 |
| Hardware | any webcam, any microphone |
| Internet | only on the first launch (models are downloaded once) |

## Installation (Windows)

```powershell
git clone https://github.com/glokmane/hand-control-x.git
cd hand-control-x

python -m venv .venv
.\.venv\Scripts\activate

pip install -r WIN\hand_control\requirements.txt
```

> On the first run the app downloads `hand_landmarker.task` (~8 MB),
> `face_landmarker.task` (~4 MB) and the Russian Vosk model (~50 MB).

## Running

All commands are run from `WIN\hand_control`.

| Script | What it does |
|---|---|
| `python hand_control.py` | **Main app** — gestures + voice control |
| `python voice_browser.py` | Voice only, no camera |
| `python darts_game.py` | 3D darts, controlled by hand |
| `python jedi_duel.py` | First-person Jedi duel |
| `python emotion_67.py` | Face emotions + the "67" meme |
| `python voice_windows.py` | Windows voice actions (used by the main app, runnable standalone) |

## Gestures

| Gesture | Action |
|---|---|
| Open palm | Move the cursor |
| Fist | "Lift the mouse" — cursor stays, reposition your hand |
| Thumb + index (OK sign) | Left click — tap to click, hold and move to drag |
| Thumb + middle | Right click |
| Thumb + ring | Double click |
| Index + middle (V sign) | Scroll joystick — the further from the point where you showed V, the faster it scrolls |

Keep the remaining fingers raised during a pinch (like an OK sign) so the pinch is not
mistaken for a fist.

**Camera window keys:** `P` — pause · `L` — pick the hand again · `Q` / `Esc` — quit.

## Voice commands

Say **"команды"** ("commands") to see the full list on screen. A short overview:

### System (Windows)

| Say | Result |
|---|---|
| `запусти телеграм` | Launch any installed app via Windows search |
| `открой проводник / блокнот / калькулятор / параметры` | Open Explorer / Notepad / Calculator / Settings |
| `открой папку загрузки / документы / рабочий стол` | Open a known folder |
| `сверни окно`, `разверни окно`, `закрой программу` | Window management |
| `окно влево` / `окно вправо`, `следующее окно` | Snap / switch windows |
| `новый рабочий стол`, `следующий рабочий стол` | Virtual desktops |
| `громкость 40`, `громкость на максимум`, `громче`, `тише` | Volume |
| `яркость 70`, `ярче`, `темнее` | Brightness |
| `который час`, `какое сегодня число`, `заряд батареи` | Spoken answers |
| `скриншот области`, `буфер обмена`, `смени язык` | Utility actions |
| `заблокируй компьютер`, `спящий режим` | Lock / sleep |
| `выключи компьютер`, `перезагрузи компьютер` | Asks for confirmation ("да"), 60 s delay, `отмени выключение` cancels |

### Browser

| Say | Result |
|---|---|
| `найди рецепт борща`, `найди на ютубе котиков` | Web / YouTube search |
| `открой ютуб`, `открой в новой вкладке википедию` | Open a site |
| `нажми войти` | Find text on the page and click it |
| `назад`, `вперёд`, `обнови`, `стоп` | Navigation |
| `новая вкладка`, `закрой вкладку`, `вкладка три` | Tabs |
| `вниз`, `вверх`, `в начало`, `в конец` | Scrolling |
| `увеличь`, `уменьши`, `полный экран`, `найди на странице ...` | Page |
| `напечатай ...`, `диктовка`, `ввод`, `выдели всё` | Text input and dictation |
| `пауза` / `играй`, `громче`, `тише`, `без звука` | Video and sound |

### Gestures and voice together

`сменить руку`, `выключи жесты` / `включи жесты`, `изменить ди-пи-ай` + a number from
400 to 2500 (e.g. `изменить ди-пи-ай тысяча`).

### Yandex Music

`следующий трек`, `предыдущий трек`, `перемотай вперёд`, `перемотай назад на 30 секунд`,
`трек сначала`, `останови музыку`, `включи музыку`, `что играет`, `открой яндекс музыку`.
Works even when the tab is in the background and the browser is minimized.

## Games

**Darts** — move an open palm to aim, pinch to pick up a dart, then swing your hand up
and release to throw. 5 rounds × 3 darts.
Keys: `M` — switch camera/mouse mode · `R` — restart · `Esc` — quit.

**Jedi duel** — right hand holds the lightsaber (fist as the hilt): tilt to angle the blade,
block the incoming strike, parry with a sharp motion, attack when the enemy is close.
Left hand uses the Force: push (palm forward), pull (palm or fist back), choke (hold an
OK sign), and disarm (hold an open palm still for ~0.6 s, then clench a fist).
Keys: `M` — camera/mouse · `X` — swap hands · `H` — strike ghost · `Tab` — help ·
`R` — restart · `Esc` — quit.

**Emotions 67** — MediaPipe Face Landmarker reads 52 blendshapes and picks an emoji
(laughing, happy, sad, surprised, angry, wink, kiss, sleeping). The first 2 seconds are
calibration — keep a neutral face. Move both hands up and down alternately to summon the
"67" meme arms. Keys: `C` — recalibrate · `Esc` / `Q` — quit.

## Project structure

```
hand-control-x/
├── WIN/
│   └── hand_control/
│       ├── hand_control.py     # main app: cursor, gestures, hand tracking
│       ├── voice_browser.py    # offline speech engine + browser commands
│       ├── voice_windows.py    # Windows-specific voice actions
│       ├── yandex_music.py     # Yandex Music via Windows media controls
│       ├── darts_game.py       # 3D darts game
│       ├── jedi_duel.py        # first-person Jedi duel
│       ├── emotion_67.py       # face emotions + "67" meme
│       └── requirements.txt
├── MAC/                        # macOS port (planned)
├── ARCHLINUX/                  # Arch Linux port (planned)
├── CONTRIBUTING.md
├── CLA.md
└── LICENSE
```

## Roadmap

- [x] Windows: gestures, voice, games
- [ ] macOS port — camera capture and accessibility permissions
- [ ] Arch Linux port — X11/Wayland input, PipeWire volume control
- [ ] Settings UI instead of JSON files
- [ ] English voice commands (Vosk EN model)
- [ ] Installer / single-file build

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot open camera` | Close other apps using the webcam (Zoom, Teams, OBS). Only one process can hold the camera. |
| Model download fails | Check your connection and proxy settings; models are fetched from `storage.googleapis.com` and `alphacephei.com`. |
| Voice commands are not recognized | Allow microphone access in Windows privacy settings, speak a bit slower, and make sure no other app is recording. |
| Cursor is too fast / slow | Say `изменить ди-пи-ай` + a number, or edit `hand_settings.json`. |
| `ModuleNotFoundError` | Activate the virtual environment and reinstall `requirements.txt`. |
| Low FPS | Close background apps; the app runs MediaPipe on the CPU. |

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.
Every contributor signs [`CLA.md`](CLA.md) so that the project stays legally clean.

## License

Proprietary — **all rights reserved**. See [`LICENSE`](LICENSE).
Commercial licensing is available on request.
