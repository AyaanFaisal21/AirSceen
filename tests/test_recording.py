from __future__ import annotations

import json

from airscreen.landmarks import HandLandmarks, Landmark
from airscreen.recording import LandmarkSessionRecorder
from airscreen.vision.camera import Frame
from airscreen.vision.gaze_tracker import GazeEstimate


def test_landmark_session_recorder_writes_frame_samples_as_jsonl(tmp_path) -> None:
    output_path = tmp_path / "session.jsonl"
    recorder = LandmarkSessionRecorder(output_path)

    recorder.record(
        frame_index=3,
        frame=Frame(width=640, height=480, data=object()),
        hands=[
            HandLandmarks(
                wrist=Landmark(0.1, 0.2, 0.3),
                thumb_tip=Landmark(0.2, 0.3, 0.4),
                index_tip=Landmark(0.4, 0.5, 0.6),
            )
        ],
        gaze_estimate=GazeEstimate(x=0.45, y=0.55, confidence=0.75),
    )
    recorder.close()

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["schema_version"] == 1
    assert payload["frame_index"] == 3
    assert payload["frame"] == {"height": 480, "width": 640}
    assert payload["hands"][0]["index_tip"] == {"x": 0.4, "y": 0.5, "z": 0.6}
    assert payload["hands"][0]["middle_tip"] is None
    assert payload["gaze"] == {"confidence": 0.75, "x": 0.45, "y": 0.55}
