"""
audio/capture.py

Realtime Audio Capture Engine
"""

import queue
import sounddevice as sd # type: ignore

from config.settings import SAMPLE_RATE
from config.settings import CHANNELS
from config.settings import BLOCK_SIZE
from config.settings import DTYPE


class AudioCapture:

    def __init__(self):

        self.audio_queue = queue.Queue()

        self.stream = None

    def callback(self, indata, frames, time, status):

        if status:
            print(status)

        self.audio_queue.put(indata.copy())

    def start(self):

        self.stream = sd.InputStream(

            samplerate=SAMPLE_RATE,

            channels=CHANNELS,

            blocksize=BLOCK_SIZE,

            dtype=DTYPE,

            callback=self.callback

        )

        self.stream.start()

        print("Microphone Started.")

    def stop(self):

        if self.stream:

            self.stream.stop()

            self.stream.close()

            print("Microphone Stopped.")

    def get_audio(self):

        if self.audio_queue.empty():

            return None

        return self.audio_queue.get()