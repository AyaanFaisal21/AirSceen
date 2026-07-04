from __future__ import annotations

from airscreen.config import AirScreenConfig


class AirScreenApp:
    def __init__(self, config: AirScreenConfig) -> None:
        self._config = config

    def run(self) -> int:
        if self._config.dry_run:
            print("AirScreen dry run: configuration loaded and runtime wiring is ready.")
            return 0

        print("AirScreen live runtime is not implemented yet. Use --dry-run for now.")
        return 2
