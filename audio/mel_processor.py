"""
audio/mel_processor.py

Mel Spectrogram Generator
"""

from __future__ import annotations

import numpy as np
import librosa # type: ignore


class MelSpectrogramProcessor:
    """
    실시간 Mel Spectrogram 생성기
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
    ) -> None:

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Audio → Mel Spectrogram
        """

        audio = np.asarray(audio, dtype=np.float32)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        return mel_db