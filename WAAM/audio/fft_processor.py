"""
audio/fft_processor.py

FFT Processor

실시간 오디오를
주파수 영역으로 변환한다.
"""

from __future__ import annotations

import numpy as np


class FFTProcessor:
    """
    Fast Fourier Transform Processor
    """

    def __init__(self, sample_rate: int):

        self.sample_rate = sample_rate

    def process(
        self,
        audio: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        FFT 수행

        Returns
        -------
        frequencies

        magnitude
        """

        audio = np.asarray(audio, dtype=np.float32)

        n = len(audio)

        fft_result = np.fft.rfft(audio)

        magnitude = np.abs(fft_result)

        frequencies = np.fft.rfftfreq(
            n,
            d=1 / self.sample_rate
        )

        return frequencies, magnitude