# macOS port — in development

This folder is reserved for the macOS version of Hand Control X. Nothing is implemented yet.

## Plan

| Item | Note |
|---|---|
| Camera capture | `cv2.VideoCapture(0)` with AVFoundation backend instead of `CAP_DSHOW` |
| Cursor and clicks | `pyautogui` / Quartz `CGEvent` — may require the Accessibility permission |
| Media keys | `osascript` or `media-control` instead of Windows SMTC |
| Volume / brightness | `osascript` for volume; brightness via `brightness` CLI or `CoreDisplay` |
| Voice input | Vosk (same model), microphone via `sounddevice` |

## Expected layout

```
MAC/
└── hand_control/
    ├── hand_control.py
    ├── voice_browser.py
    ├── voice_macos.py
    └── requirements.txt
```

## Permissions the app will need

- **Camera** — System Settings → Privacy & Security → Camera
- **Microphone** — Privacy & Security → Microphone
- **Accessibility** — Privacy & Security → Accessibility (for synthetic mouse events)

## Contributing

Read [`../CONTRIBUTING.md`](../CONTRIBUTING.md) and sign [`../CLA.md`](../CLA.md) first.
Platform-specific code must not break the Windows build.
