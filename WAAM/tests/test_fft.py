import numpy as np

from audio.fft_processor import FFTProcessor


sample_rate = 44100

processor = FFTProcessor(sample_rate)


duration = 1

t = np.linspace(
    0,
    duration,
    sample_rate,
    endpoint=False
)

audio = np.sin(
    2 * np.pi * 1000 * t
)

freq, mag = processor.process(audio)

peak = np.argmax(mag)

print()

print("Peak Frequency")

print(freq[peak])