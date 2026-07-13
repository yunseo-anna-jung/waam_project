import numpy as np

from audio.mel_processor import MelSpectrogramProcessor


processor = MelSpectrogramProcessor(
    sample_rate=44100
)

t = np.linspace(
    0,
    2,
    44100 * 2,
    endpoint=False
)

audio = np.sin(
    2*np.pi*1000*t
)

mel = processor.process(audio)

print(mel.shape)

print(mel.dtype)

print(mel.min())

print(mel.max())