"""
dataset/session.py

WAAM Dataset Session Manager

데이터 저장 주기를 제어한다.
"""

from __future__ import annotations

import time
from typing import Optional


class DatasetSession:
    """
    Dataset Recording Session
    """

    def __init__(
        self,
        save_interval: float = 1.0,
    ) -> None:

        self.save_interval = save_interval

        self.last_save_time = 0.0

        self.total_saved = 0

        self.is_running = False

    def start(self) -> None:

        self.is_running = True

        self.last_save_time = 0.0

        self.total_saved = 0

        print("Dataset Session Started.")

    def stop(self) -> None:

        self.is_running = False

        print("Dataset Session Finished.")

    def should_save(self) -> bool:

        if not self.is_running:

            return False

        current = time.time()

        if current - self.last_save_time >= self.save_interval:

            self.last_save_time = current

            self.total_saved += 1

            return True

        return False

    def saved_count(self) -> int:

        return self.total_saved

    def running(self) -> bool:

        return self.is_running