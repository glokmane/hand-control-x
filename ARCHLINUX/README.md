# Arch Linux port — in development

This folder is reserved for the Arch Linux version of Hand Control X. Nothing is implemented yet.

## Plan

| Item | Note |
|---|---|
| Camera capture | `cv2.VideoCapture(0)` with V4L2 (`CAP_V4L2`) instead of `CAP_DSHOW` |
| Cursor and clicks | `python-xlib` (X11) or `ydotool` / `wtype` (Wayland) |
| Volume | `pactl` / `wpctl` (PipeWire) |
| Brightness | `brightnessctl` or direct `/sys/class/backlight` |
| Media keys | `playerctl` instead of Windows SMTC |
| Voice input | Vosk model + `sounddevice`, no Windows-specific code |
| Notifications | `notify-send` |

## Planned packages

```bash
sudo pacman -S python python-pip tk mesa libgl v4l-utils pipewire pipewire-pulse \
               playerctl brightnessctl ydotool wtype xdotool
pip install -r ARCHLINUX/hand_control/requirements.txt
```

`mediapipe` wheels exist for `x86_64`; on other architectures you may need a container.

## Preview (no plan yet)

```
ARCHLINUX/
└── hand_control/
    ├── hand_control.py
    ├── voice_browser.py
    ├── voice_arch.py
    └── requirements.txt
```

## Contributing

Read [`../CONTRIBUTING.md`](../CONTRIBUTING.md) and sign [`../CLA.md`](../CLA.md) first.
Both X11 and Wayland must be considered — do not assume one of them in shared code.
