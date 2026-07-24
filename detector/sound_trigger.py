"""
detector/sound_trigger.py

WAAM Acoustic Change Detector

현재 Mel Spectrogram과
이전 Mel Spectrogram을 비교하여
Change Score를 계산한다.

이 모듈은 의사결정을 하지 않는다.
오직 Change Score만 계산한다.
"""

from __future__ import annotations

import numpy as np


class SoundTrigger:
    """
    Acoustic Change Score Calculator
    """

    def __init__(self) -> None:

        # 이전 Mel Spectrogram
        self.previous_mel: np.ndarray | None = None

    def reset(self) -> None:
        """
        이전 데이터를 초기화한다.
        """
        self.previous_mel = None

    def update(self, current_mel: np.ndarray) -> float:
        """
        현재 Mel과 이전 Mel의 차이를 계산한다.

        Parameters
        ----------
        current_mel : np.ndarray
            현재 Mel Spectrogram

        Returns
        -------
        float
            Change Score
        """

        current_mel = np.asarray(current_mel, dtype=np.float32)

        # 첫 번째 프레임은 비교 대상이 없으므로
        # 기준 데이터로 저장하고 Score=0 반환
        if self.previous_mel is None:

            self.previous_mel = current_mel.copy()

            return 0.0

        # Mean Absolute Difference
        score = float(

            np.mean(

                np.abs(

                    current_mel - self.previous_mel

                )

            )

        )

        # 다음 비교를 위해 현재 Mel 저장
        self.previous_mel = current_mel.copy()

        return score