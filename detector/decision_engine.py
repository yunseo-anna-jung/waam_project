"""
detector/decision_engine.py

WAAM Decision Engine

실시간 음향 분석 결과를 바탕으로
현재 상태를 판정하는 모듈.

이 모듈은 Score를 계산하지 않는다.
Score를 입력받아 시스템의 Action만 결정한다.

Author : Yunseo Anna Jung
"""

from __future__ import annotations

from dataclasses import dataclass
import time


# ==========================================================
# Decision Result
# ==========================================================

@dataclass(slots=True)
class DecisionAction:
    """
    DecisionEngine의 판단 결과

    Attributes
    ----------
    save : bool
        현재 데이터를 저장할 것인가

    warning : bool
        경고를 발생시킬 것인가

    label : str
        데이터셋 라벨
        ("normal", "warning", "danger")

    level : str
        화면 표시용 상태

    score : float
        Filtered Change Score

    timestamp : float
        UNIX Timestamp

    frame_id : int
        현재 프레임 번호
    """

    save: bool

    warning: bool

    label: str

    level: str

    score: float

    timestamp: float

    frame_id: int


# ==========================================================
# Decision Engine
# ==========================================================

class DecisionEngine:
    """
    WAAM Decision Engine

    Filtered Score를 입력받아
    현재 시스템 상태를 판정한다.
    """

    def __init__(

        self,

        warning_threshold: float,

        danger_threshold: float,

    ) -> None:

        if warning_threshold >= danger_threshold:

            raise ValueError(

                "warning_threshold must be smaller than danger_threshold."

            )

        self.warning_threshold = warning_threshold

        self.danger_threshold = danger_threshold

        self.frame_id = 0

    # ------------------------------------------------------

    def reset(self) -> None:
        """
        프레임 번호 초기화
        """

        self.frame_id = 0

    # ------------------------------------------------------

    def update(

        self,

        score: float,

    ) -> DecisionAction:
        """
        Parameters
        ----------
        score : float
            Filtered Change Score

        Returns
        -------
        DecisionAction
        """

        self.frame_id += 1

        timestamp = time.time()

        score = float(score)

        # ==========================================
        # NORMAL
        # ==========================================

        if score < self.warning_threshold:

            return DecisionAction(

                save=False,

                warning=False,

                label="normal",

                level="NORMAL",

                score=score,

                timestamp=timestamp,

                frame_id=self.frame_id,

            )

        # ==========================================
        # WARNING
        # ==========================================

        if score < self.danger_threshold:

            return DecisionAction(

                save=True,

                warning=True,

                label="warning",

                level="WARNING",

                score=score,

                timestamp=timestamp,

                frame_id=self.frame_id,

            )

        # ==========================================
        # DANGER
        # ==========================================

        return DecisionAction(

            save=True,

            warning=True,

            label="danger",

            level="DANGER",

            score=score,

            timestamp=timestamp,

            frame_id=self.frame_id,

        )