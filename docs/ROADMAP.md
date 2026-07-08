# Roadmap

## Phase 0: Scaffold

- [x] Python package and CLI.
- [x] Runtime configuration object.
- [x] Interfaces for camera, hand tracking, gaze tracking, and pointer control.
- [x] Deterministic pinch click detector with tests.
- [x] Project map and implementation roadmap.

## Phase 1: Hand Tracking Prototype

- [x] Add OpenCV camera frame source.
- [x] Add MediaPipe hand tracker adapter.
- [x] Add debug preview with landmark overlay.
- [x] Record short landmark sessions for testing and calibration.

## Phase 2: Pointer Movement

- [ ] Map normalized fingertip coordinates to screen coordinates.
- [ ] Add smoothing and low-confidence suppression.
- [ ] Add visible enable/disable state.
- [ ] Add real pointer adapter behind an explicit `--enable-pointer` flag.

## Phase 3: Click Gesture

- [ ] Tune pinch thresholds from recorded samples.
- [ ] Add drag-ready state machine for press, hold, release.
- [ ] Add per-user gesture calibration.
- [ ] Add false-positive tests with non-click motion samples.

## Phase 4: Gaze Assist

- [x] Add MediaPipe face mesh based gaze estimation.
- [x] Add guided target set and reusable gaze calibration profile mapping.
- [ ] Add guided calibration capture UI.
- [ ] Combine gaze and hand input for target disambiguation.
- [ ] Keep gaze optional until accuracy is useful.

## Phase 5: Learned Gestures

- [ ] Define capture format for labeled gesture samples.
- [ ] Train a lightweight classifier for click and hold gestures.
- [ ] Version model metadata and evaluation reports.
- [ ] Fall back to geometry-based pinch detection when model confidence is low.
