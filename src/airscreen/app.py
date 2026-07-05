from __future__ import annotations

from airscreen.config import AirScreenConfig
from airscreen.vision.preview import DebugPreviewRunner


class AirScreenApp:
    def __init__(self, config: AirScreenConfig) -> None:
        self._config = config

    def run(self) -> int:
        if self._config.dry_run:
            print("AirScreen dry run: configuration loaded and runtime wiring is ready.")
            return 0

        if self._config.debug_preview:
            try:
                return DebugPreviewRunner(self._config).run()
            except RuntimeError as exc:
                print(f"AirScreen debug preview error: {exc}")
                return 2

        print("AirScreen live runtime is not implemented yet. Use --dry-run for now.")
        return 2
