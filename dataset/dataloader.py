"""
dataset/dataloader.py

WAAM Dataset Loader

Author : Yunseo Anna Jung

Recorder가 저장한 Mel Spectrogram을
TensorFlow 학습용 Dataset으로 변환한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split


class DatasetLoader:
    """
    WAAM Dataset Loader

    기능
    -------------------------
    1. metadata.csv 읽기
    2. Mel Spectrogram 로드
    3. Normalization
    4. CNN 입력 Shape 생성
    5. Label Encoding
    6. TensorFlow Dataset 생성
    """

    LABEL_MAP = {
        "normal": 0,
        "warning": 1,
        "danger": 2,
    }

    REVERSE_LABEL_MAP = {
        0: "normal",
        1: "warning",
        2: "danger",
    }

    # =====================================================

    def __init__(
        self,
        metadata_path: str | Path,
        dataset_root: str | Path,
    ):

        self.metadata_path = Path(metadata_path)
        self.dataset_root = Path(dataset_root)

        self.metadata = pd.read_csv(self.metadata_path)

    # =====================================================

    def _load_numpy(
        self,
        filename: str,
        label: str,
    ) -> np.ndarray:
        """
        npy 파일 로드
        """

        filepath = (
            self.dataset_root
            / label
            / filename
        )

        mel = np.load(filepath)

        return mel.astype(np.float32)

    # =====================================================

    def preprocess(
        self,
        mel: np.ndarray,
    ) -> np.ndarray:
        """
        Mel Spectrogram 전처리

        1. float32
        2. Min-Max Normalization
        3. CNN Shape
        """

        mel = mel.astype(np.float32)

        mel_min = mel.min()
        mel_max = mel.max()

        mel = (
            mel - mel_min
        ) / (
            mel_max - mel_min + 1e-8
        )

        if mel.ndim == 2:
            mel = np.expand_dims(
                mel,
                axis=-1,
            )

        return mel

    # =====================================================

    def preprocess_single(
        self,
        mel: np.ndarray,
    ) -> np.ndarray:
        """
        실시간 추론용 전처리
        """

        mel = self.preprocess(mel)

        mel = np.expand_dims(
            mel,
            axis=0,
        )

        return mel

    # =====================================================

    def load(self):
        """
        전체 데이터 로드

        CNN 학습용
        """

        x = []
        y = []

        for _, row in self.metadata.iterrows():

            mel = self._load_numpy(
                row["filename"],
                row["label"],
            )

            mel = self.preprocess(mel)

            x.append(mel)

            y.append(
                self.LABEL_MAP[
                    row["label"]
                ]
            )

        x = np.asarray(
            x,
            dtype=np.float32,
        )

        y = np.asarray(
            y,
            dtype=np.int32,
        )

        return x, y

    # =====================================================

    def load_normal(self):
        """
        정상 데이터만 로드

        AutoEncoder 학습용
        """

        x = []

        rows = self.metadata[
            self.metadata["label"] == "normal"
        ]

        for _, row in rows.iterrows():

            mel = self._load_numpy(
                row["filename"],
                row["label"],
            )

            mel = self.preprocess(mel)

            x.append(mel)

        return np.asarray(
            x,
            dtype=np.float32,
        )

    # =====================================================

    def train_validation_split(
        self,
        test_size=0.2,
        random_state=42,
    ):

        x, y = self.load()

        return train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    # =====================================================

    def to_tf_dataset(
        self,
        batch_size=32,
        shuffle=True,
    ):
        """
        CNN 학습용 Dataset
        """

        x, y = self.load()

        dataset = tf.data.Dataset.from_tensor_slices(
            (x, y)
        )

        if shuffle:
            dataset = dataset.shuffle(
                len(x)
            )

        dataset = dataset.cache()

        dataset = dataset.batch(
            batch_size
        )

        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )

        return dataset

    # =====================================================

    def to_autoencoder_dataset(
        self,
        batch_size=32,
        shuffle=True,
    ):
        """
        AutoEncoder 학습용 Dataset

        입력 = 출력
        """

        x = self.load_normal()

        dataset = tf.data.Dataset.from_tensor_slices(
            (x, x)
        )

        if shuffle:
            dataset = dataset.shuffle(
                len(x)
            )

        dataset = dataset.cache()

        dataset = dataset.batch(
            batch_size
        )

        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )

        return dataset