"""
detector/score_filter.py

WAAM Score Filter

Change Score를 실시간으로 평활화(EMA)하는 모듈.

Author : Yunseo Anna Jung
"""

from __future__ import annotations


class ScoreFilter:
    """
    Exponential Moving Average Filter
    """

    def __init__(self, alpha: float = 0.2) -> None:
        """
        Parameters
        ----------
        alpha : float
            EMA 계수
            0에 가까울수록 더 부드럽게 반응
            1에 가까울수록 원본 점수에 가깝게 반응
        """

        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1.")

        self.alpha = alpha
        self._ema: float | None = None

    def reset(self) -> None:
        """
        필터 상태 초기화
        """
        self._ema = None

    def update(self, score: float) -> float:
        """
        새로운 Change Score를 입력받아
        EMA를 계산하여 반환한다.

        Parameters
        ----------
        score : float
            Raw Change Score

        Returns
        -------
        float
            Filtered Score
        """

        score = float(score)

        if self._ema is None:
            self._ema = score
        else:
            self._ema = (
                self.alpha * score
                + (1.0 - self.alpha) * self._ema
            )

        return self._ema

    @property
    def value(self) -> float:
        """
        현재 EMA 값 반환
        """

        if self._ema is None:
            return 0.0

        return self._ema