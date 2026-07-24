"""
core/pipeline.py

WAAM Real-Time Audio Processing Pipeline

Author : Yunseo Anna Jung

모든 실시간 모듈을 연결하는 메인 파이프라인
"""

from __future__ import annotations

import time

# ===============================
# Config
# ===============================

from config.settings import ( # type: ignore
    PIPELINE_INTERVAL,
    SAMPLE_RATE,
    BUFFER_DURATION,
    SAVE_INTERVAL,
    CHANGE_SCORE_ALPHA,
    WARNING_THRESHOLD,
    DANGER_THRESHOLD,
    PRINT_SCORE,
    PRINT_SAVE,
)

# ===============================
# Audio
# ===============================

from audio.capture import AudioCapture
from audio.buffer import AudioBuffer
from audio.mel_processor import MelSpectrogramProcessor

# ===============================
# Detector
# ===============================

from detector.sound_trigger import SoundTrigger
from detector.score_filter import ScoreFilter # type: ignore
from detector.decision_engine import DecisionEngine

# ===============================
# Dataset
# ===============================

from dataset.session import DatasetSession
from dataset.recorder import DatasetRecorder


class WAAMPipeline:
    """
    WAAM Real-Time Pipeline
    """

    def __init__(self):

        # -------------------------
        # Audio
        # -------------------------

        self.capture = AudioCapture()

        self.buffer = AudioBuffer(
            sample_rate=SAMPLE_RATE,
            duration=BUFFER_DURATION,
        )

        self.mel = MelSpectrogramProcessor(
            sample_rate=SAMPLE_RATE,
        )

        # -------------------------
        # Detector
        # -------------------------

        self.trigger = SoundTrigger()

        self.score_filter = ScoreFilter(
            alpha=CHANGE_SCORE_ALPHA
        )

        self.decision = DecisionEngine(
            warning_threshold=WARNING_THRESHOLD,
            danger_threshold=DANGER_THRESHOLD,
        )

        # -------------------------
        # Dataset
        # -------------------------

        self.session = DatasetSession(
            save_interval=SAVE_INTERVAL
        )

        self.recorder = DatasetRecorder()

    # ===========================================

    def start(self):

        print("=" * 60)
        print(" WAAM Real-Time Monitoring ")
        print("=" * 60)

        print(f"Sample Rate : {SAMPLE_RATE}")
        print(f"Buffer      : {BUFFER_DURATION}s")
        print("=" * 60)

        self.capture.start()
        self.session.start()

    # ===========================================

    def stop(self):

        self.capture.stop()
        self.session.stop()

        print()

        print("=" * 60)

        print("Pipeline Finished")

        print(
            f"Processed Frames : {self.decision.frame_id}"
        )

        print("=" * 60)

    # ===========================================

    def update(self):

        # ----------------------------------------
        # 1. Read Audio
        # ----------------------------------------

        try:

            audio = self.capture.read()

        except Exception as e:

            print(e)

            return

        if audio is None:
            return

        # ----------------------------------------
        # 2. Update Buffer
        # ----------------------------------------

        self.buffer.append(audio)

        if not self.buffer.is_full():
            return

        # ----------------------------------------
        # 3. Get Recent Audio
        # ----------------------------------------

        recent_audio = self.buffer.get_buffer()

        # ----------------------------------------
        # 4. Generate Mel Spectrogram
        # ----------------------------------------

        mel = self.mel.process(recent_audio)

        # ----------------------------------------
        # 5. Calculate Raw Change Score
        # ----------------------------------------

        raw_score = self.trigger.update(mel)

        # ----------------------------------------
        # 6. Smooth Score
        # ----------------------------------------

        filtered_score = self.score_filter.update(
            raw_score
        )

        # ----------------------------------------
        # 7. Decision
        # ----------------------------------------

        action = self.decision.update(
            filtered_score
        )

        # ----------------------------------------
        # 8. Print Status
        # ----------------------------------------

        if PRINT_SCORE:

            print(
                f"[Frame {action.frame_id:06d}] "
                f"[{action.level}] "
                f"Score={action.score:.5f}"
            )
            print(
                f"[Frame {action.frame_id:06d}] "
                f"{action.timestamp:.3f} "
                f"{action.level} "
                f"{action.score:.5f}"
            )

        # ----------------------------------------
        # 9. Save Dataset
        # ----------------------------------------

        if action.save and self.session.should_save():

            path = self.recorder.save(
                mel,
                label=action.label,
                score=action.score,
                timestamp=action.timestamp,
                frame_id=action.frame_id,
            )

            if PRINT_SAVE:
                print(f"Saved : {path}")

    # ===========================================

    def run(self):

        self.start()

        try:

            while True:

                self.update()

                time.sleep(PIPELINE_INTERVAL)

        except KeyboardInterrupt:

            print("\nKeyboard Interrupt")

        finally:

            self.stop()