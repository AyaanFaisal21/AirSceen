# AirScreen Project Map

## Vision

AirScreen turns webcam landmarks into desktop intent. The first usable prototype should let a user move the pointer with their index finger and click by pinching middle finger and thumb together. Gaze tracking is a secondary signal for target selection and correction, not a dependency for the first hand-control milestone.

## Runtime Data Flow

1. `FrameSource` captures frames from the selected camera.
2. `HandTracker` extracts hand landmarks for visible hands.
3. `GazeTracker` optionally estimates the user's screen gaze point.
4. Gesture detectors convert landmark movement into intent events.
5. Calibration maps normalized camera coordinates to screen coordinates.
6. Pointer controller applies movement and click actions.
7. Safety layer can pause pointer control when confidence is low or a kill switch is triggered.

## Core Modules

- Capture: camera access, frame timing, preview, and cleanup.
- Tracking: MediaPipe hand and face mesh adapters behind local interfaces.
- Gesture Recognition: pinch detection, hysteresis, dwell timers, learned gesture models.
- Calibration: screen bounds, camera orientation, user-specific correction profiles.
- Pointer Control: dry-run pointer sink first, then OS automation adapter.
- Runtime App: event loop, state machine, safety controls, debug overlay.

## Interaction Model

- Move: index fingertip controls the pointer after calibration and smoothing.
- Click: middle-thumb pinch emits one click on gesture entry, then waits for release.
- Hold: future extension where a sustained pinch becomes drag.
- Gaze Assist: future extension where gaze picks the likely target and hand motion confirms.

## Important Engineering Constraints

- Keep deterministic gesture logic testable without camera dependencies.
- Keep OS automation behind an adapter so tests never move the real pointer.
- Default to dry-run until calibration and a kill switch exist.
- Treat eye tracking as experimental because webcam-only gaze can be noisy.
- Store trained models and captures outside git unless intentionally versioned.

## Milestones

1. Package scaffold, CLI, docs, and deterministic gesture tests.
2. Live webcam preview with hand landmark overlay.
3. Finger-to-pointer mapping in dry-run/debug overlay.
4. Pointer control behind explicit opt-in safety controls.
5. Pinch click calibration and tests against recorded landmark samples.
6. Experimental gaze estimate overlay.
7. User calibration profiles and gesture training pipeline.
