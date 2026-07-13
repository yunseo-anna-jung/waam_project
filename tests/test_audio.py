import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from audio.capture import AudioCapture

capture = AudioCapture()

capture.start()

while True:

    audio = capture.get_audio()

    if audio is None:
        continue

    print(audio.shape)