"""
Global Configuration
MetalPrinterAI
"""

from pathlib import Path

# =====================================================
# Project Root
# =====================================================

ROOT_DIR = Path(__file__).parent

# =====================================================
# Dataset
# =====================================================

RAW_DATASET = ROOT_DIR / "dataset" / "raw"
SPEC_DATASET = ROOT_DIR / "dataset" / "spectrogram"
PROC_DATASET = ROOT_DIR / "dataset" / "processed"

# =====================================================
# Audio
# =====================================================

SAMPLE_RATE = 44100

CHANNELS = 1

BLOCK_SIZE = 4096

DTYPE = "float32"

# =====================================================
# AI
# =====================================================

IMAGE_SIZE = 128

LATENT_DIM = 64

THRESHOLD = 0.015

# =====================================================
# Serial
# =====================================================

BAUDRATE = 115200

TIMEOUT = 1

# =====================================================
# Logging
# =====================================================

LOG_DIR = ROOT_DIR / "logs"

LOG_FILE = LOG_DIR / "system.log"