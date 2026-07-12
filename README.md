# AirScreen

AirScreen is an experimental computer vision interface for controlling a computer with hand and eye movement. The goal is a Vision Pro-inspired interaction model on a standard desktop: hand landmarks move and click the pointer, while gaze tracking becomes an optional intent signal as the model matures.

The current repository is an initial project scaffold. It separates camera capture, hand tracking, gaze tracking, gesture recognition, calibration, and pointer control so each part can be tested and replaced independently.

## Product Direction

- Track fingertip position from webcam hand landmarks.
- Move the desktop pointer from a smoothed index-finger target.
- Detect a learned middle-thumb pinch gesture for click intent.
- Add experimental eye tracking for gaze-assisted pointer targeting.
- Support calibration profiles so tracking can adapt to a user, camera, and screen.

## Repository Map

- `src/airscreen/app.py` - application orchestration and runtime loop placeholder.
- `src/airscreen/config.py` - typed runtime configuration.
- `src/airscreen/vision/camera.py` - camera frame source abstraction.
- `src/airscreen/vision/hand_tracker.py` - hand landmark tracker interface.
- `src/airscreen/vision/gaze_tracker.py` - gaze tracker interface.
- `src/airscreen/gestures.py` - pinch/click gesture recognition primitives.
- `src/airscreen/input/pointer.py` - pointer movement and click abstraction.
- `docs/PROJECT_MAP.md` - architecture, data flow, milestones, and implementation notes.
- `docs/ROADMAP.md` - staged build plan for the prototype.
- `tests/` - focused unit tests for deterministic logic.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
airscreen --dry-run
```

The dry-run mode does not access the webcam or move the pointer yet. It verifies that the package and CLI wiring are installed correctly.

## Camera Debug Preview

Install the vision extras, then run the temporary debug preview:

```bash
python -m pip install -e ".[vision,dev]"
airscreen --debug-preview
```

The preview opens the selected camera, draws boxes and labels around detected fingertips, and shows a pinch-state indicator. Press `q` or Escape to quit. On macOS, the terminal or app running AirScreen may need Camera permission before frames appear.

Gaze tracking can be toggled into the same preview:

```bash
airscreen --debug-preview --enable-gaze
```

This draws an approximate gaze marker from MediaPipe Face Mesh/iris landmarks. It is useful for early visual validation, but it is not calibrated screen gaze yet.

Saved gaze calibration profiles can be applied to the debug preview:

```bash
airscreen --debug-preview --enable-gaze --gaze-profile .airscreen/gaze-profile.json
```

The current calibration layer fits a simple normalized X/Y correction from guided target samples. It is the deterministic profile format and mapping step; a full guided capture UI is still planned.

To capture landmark samples for later calibration or regression tests, record a short JSON Lines session from the debug preview:

```bash
airscreen --debug-preview --enable-gaze --record-landmarks captures/session.jsonl
```

Each line contains one frame sample with frame dimensions, hand landmarks, and the optional gaze estimate. Runtime captures are ignored by git by default.

## Red Circle Target Experiment

A toggleable side-project overlay can spawn red circle targets in the debug preview:

```bash
airscreen --debug-preview --red-circle-targets
```

Targets spawn sporadically, no faster than every two seconds, no slower than every ten seconds while the overlay is below its cap, and at most four are visible at once. Green targets add one point when popped or sliced. Red bad targets can appear in place of green targets, expire after a few seconds, and deduct one point if hit. Slow movement onto a target does not count as a slice.

## Planned Runtime Stack

- Python 3.11+
- OpenCV for camera frames and preview windows
- MediaPipe for hand and face landmarks
- PyAutoGUI or pynput for desktop pointer control
- NumPy for vector math and smoothing
- pytest for deterministic gesture and calibration tests

## Safety Notes

Pointer automation can interfere with normal computer use. The runtime should always keep a visible kill switch, dry-run mode, conservative default sensitivity, and calibration boundaries before enabling live pointer control.
