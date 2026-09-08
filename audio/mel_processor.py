"""
audio/mel_processor.py

Realtime Mel Spectrogram Processor
Author : Yunseo Anna Jung
"""

from __future__ import annotations

import librosa
import numpy as np
import cv2


class MelSpectrogramProcessor:

    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        image_size: int = 128,
    ) -> None:

        self.sample_rate = sample_rate

        self.n_fft = n_fft

        self.hop_length = hop_length

        self.n_mels = n_mels

        self.image_size = image_size

    def process(
        self,
        audio: np.ndarray
    ) -> np.ndarray:

        audio = np.asarray(audio, dtype=np.float32)

        # ----------------------------
        # Mel Spectrogram
        # ----------------------------

        mel = librosa.feature.melspectrogram(

            y=audio,

            sr=self.sample_rate,

            n_fft=self.n_fft,

            hop_length=self.hop_length,

            n_mels=self.n_mels,

            power=2.0

        )

        # ----------------------------
        # dB 변환
        # ----------------------------

        mel = librosa.power_to_db(

            mel,

            ref=np.max

        )

        # ----------------------------
        # Resize
        # ----------------------------

        mel = cv2.resize(

            mel,

            (self.image_size, self.image_size),

            interpolation=cv2.INTER_AREA

        )

        # ----------------------------
        # Normalize
        # ----------------------------

        mel -= mel.min()

        mel /= (mel.max() + 1e-8)

        mel = mel.astype(np.float32)

        # ----------------------------
        # Channel 추가
        # ----------------------------

        mel = np.expand_dims(

            mel,

            axis=-1

        )

        return mel