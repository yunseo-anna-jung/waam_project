"""
dataset/recorder.py

WAAM Dataset Recorder

Mel Spectrogram과 메타데이터를 저장하는 모듈

Author : Yunseo Anna Jung
"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from config.settings import (
    DATASET_DIR,
    NORMAL_DIR,
    WARNING_DIR,
)

# danger 폴더가 settings.py에 없다면 기본 경로 사용
try:
    from config.settings import DANGER_DIR
except ImportError:
    DANGER_DIR = Path(DATASET_DIR) / "danger"


class DatasetRecorder:
    """
    WAAM Dataset Recorder

    저장 구조

    dataset/
        normal/
        warning/
        danger/
        metadata.csv
    """

    def __init__(self) -> None:

        self.dataset_dir = Path(DATASET_DIR)

        self.normal_dir = Path(NORMAL_DIR)
        self.warning_dir = Path(WARNING_DIR)
        self.danger_dir = Path(DANGER_DIR)

        self.metadata_path = (
            self.dataset_dir / "metadata.csv"
        )

        self._create_directories()
        self._initialize_metadata()

    # =====================================================
    # Directory
    # =====================================================

    def _create_directories(self) -> None:
        """필요한 데이터셋 폴더 생성"""

        self.normal_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.warning_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.danger_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =====================================================
    # Metadata
    # =====================================================

    def _initialize_metadata(self) -> None:
        """
        metadata.csv 생성

        Session ID까지 기록한다.
        """

        if self.metadata_path.exists():
            return

        with open(
            self.metadata_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "frame_id",
                "timestamp",
                "score",
                "label",
                "filename",
                "session_id",
            ])

    # =====================================================
    # Directory Selection
    # =====================================================

    def _select_directory(
        self,
        label: str,
    ) -> Path:
        """Label에 따라 저장 폴더 선택"""

        label = label.lower()

        if label == "normal":
            return self.normal_dir

        if label == "warning":
            return self.warning_dir

        if label == "danger":
            return self.danger_dir

        raise ValueError(
            f"Unknown label : {label}"
        )

    # =====================================================
    # Save
    # =====================================================

    def save(
        self,
        mel: np.ndarray,
        label: str,
        score: float,
        timestamp: float,
        frame_id: int,
        session_id: str,
    ) -> Path:
        """
        Mel Spectrogram과 메타데이터 저장

        Parameters
        ----------
        mel : np.ndarray
            Mel Spectrogram

        label : str
            normal / warning / danger

        score : float
            Filtered Change Score

        timestamp : float
            UNIX Timestamp

        frame_id : int
            Pipeline Frame ID

        session_id : str
            현재 Dataset Session ID

        Returns
        -------
        Path
            저장된 .npy 파일 경로
        """

        if not session_id:
            raise ValueError(
                "session_id가 비어 있습니다."
            )

        directory = self._select_directory(
            label
        )

        filename = (
            f"frame_{frame_id:06d}.npy"
        )

        filepath = directory / filename

        # -----------------------------------------
        # 1. Mel Spectrogram 저장
        # -----------------------------------------

        np.save(
            filepath,
            mel,
        )

        # -----------------------------------------
        # 2. Metadata 저장
        # -----------------------------------------

        with open(
            self.metadata_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                frame_id,
                timestamp,
                score,
                label,
                filename,
                session_id,
            ])

        return filepath