"""
audio/buffer.py

Realtime Ring Buffer
Author : Yunseo Anna Jung

최근 일정 시간의 오디오 데이터를 유지하는 버퍼
"""

from collections import deque
from typing import Optional

import numpy as np


class AudioBuffer:
    """
    실시간 오디오 Ring Buffer

    Parameters
    ----------
    sample_rate : int
        샘플링 주파수 (Hz)

    duration : float
        버퍼 길이 (초)

    channels : int
        오디오 채널 수
    """

    def __init__(
        self,
        sample_rate: int,
        duration: float,
        channels: int = 1,
    ) -> None:

        self.sample_rate = sample_rate
        self.duration = duration
        self.channels = channels

        self.max_samples = int(sample_rate * duration)

        self.buffer = deque(maxlen=self.max_samples)

    def append(self, audio: np.ndarray) -> None:
        """
        새로운 오디오 데이터를 버퍼에 저장한다.
        """

        audio = np.asarray(audio)

        if audio.ndim == 2:
            audio = audio[:, 0]

        self.buffer.extend(audio)

    def get_buffer(self) -> np.ndarray:
        """
        현재 버퍼 전체를 반환한다.
        """

        if len(self.buffer) == 0:
            return np.array([], dtype=np.float32)

        return np.array(self.buffer, dtype=np.float32)

    def is_full(self) -> bool:
        """
        버퍼가 가득 찼는지 확인
        """

        return len(self.buffer) >= self.max_samples

    def clear(self) -> None:
        """
        버퍼 초기화
        """

        self.buffer.clear()

    def current_length(self) -> int:
        """
        현재 저장된 샘플 수
        """

        return len(self.buffer)

    def seconds(self) -> float:
        """
        현재 저장된 시간(초)
        """

        return len(self.buffer) / self.sample_rate