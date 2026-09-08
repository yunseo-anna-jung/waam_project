# tests/test_trigger.py

import numpy as np

from detector.sound_trigger import SoundTrigger


trigger = SoundTrigger()

mel1 = np.random.rand(128,128,1).astype(np.float32)

mel2 = mel1.copy()

mel3 = mel1 + 0.2


print(trigger.update(mel1))

print(trigger.update(mel2))

print(trigger.update(mel3))