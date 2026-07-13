import numpy as np

from audio.mel_processor import MelSpectrogramProcessor


sample_rate = 44100

processor = MelSpectrogramProcessor(sample_rate)

duration = 2

t = np.linspace(
    0,
    duration,
    sample_rate * duration,
    endpoint=False
)

audio = np.sin(
    2 * np.pi * 1000 * t
)

mel = processor.process(audio)

print(mel.shape)

print(mel.min())

print(mel.max())