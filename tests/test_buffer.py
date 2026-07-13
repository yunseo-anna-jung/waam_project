import numpy as np

from audio.buffer import AudioBuffer


buffer = AudioBuffer(
    sample_rate=44100,
    duration=2,
)

for _ in range(50):

    fake_audio = np.random.randn(4096)

    buffer.append(fake_audio)

    print(
        buffer.current_length(),
        f"{buffer.seconds():.2f} sec"
    )

print()

print(buffer.is_full())

print(buffer.get_buffer().shape)