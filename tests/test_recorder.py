import numpy as np

from dataset.recorder import DatasetRecorder


recorder = DatasetRecorder()

dummy = np.random.rand(
    128,
    128,
    1
).astype(np.float32)

path = recorder.save(

    dummy,

    label="normal"

)

print(path)