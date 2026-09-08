"""
ai/train_autoencoder.py

WAAM AutoEncoder Training

Author : Yunseo Anna Jung

정상 상태의 Mel Spectrogram만 사용하여
Convolutional AutoEncoder를 학습한다.

학습 결과
-------------------------
1. AutoEncoder 모델
2. 학습 History
3. Best Model Checkpoint

저장 위치
-------------------------
ai/models/autoencoder.keras
ai/models/autoencoder_best.keras
ai/history/autoencoder_history.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from dataset.dataloader import DatasetLoader
from ai.autoencoder import AutoEncoder


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT / "dataset"
)

METADATA_PATH = (
    DATASET_ROOT / "metadata.csv"
)

MODEL_DIR = (
    PROJECT_ROOT / "ai" / "models"
)

HISTORY_DIR = (
    PROJECT_ROOT / "ai" / "history"
)

MODEL_PATH = (
    MODEL_DIR / "autoencoder.keras"
)

BEST_MODEL_PATH = (
    MODEL_DIR / "autoencoder_best.keras"
)

HISTORY_PATH = (
    HISTORY_DIR / "autoencoder_history.csv"
)


# ============================================================
# Training Configuration
# ============================================================

BATCH_SIZE = 32

EPOCHS = 100

VALIDATION_SPLIT = 0.2

RANDOM_STATE = 42

LEARNING_RATE = 1e-3

PATIENCE = 10


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(RANDOM_STATE)

tf.random.set_seed(RANDOM_STATE)


# ============================================================
# Directory Preparation
# ============================================================

def prepare_directories():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Dataset Loading
# ============================================================

def load_dataset():

    print("=" * 60)
    print("Loading WAAM Dataset")
    print("=" * 60)

    loader = DatasetLoader(
        metadata_path=METADATA_PATH,
        dataset_root=DATASET_ROOT,
    )

    # -----------------------------------------
    # AutoEncoder는 정상 데이터만 사용
    # -----------------------------------------

    x = loader.load_normal()

    if len(x) == 0:

        raise RuntimeError(
            "정상(normal) 데이터가 없습니다."
        )

    print(
        f"Normal samples : {len(x)}"
    )

    print(
        f"Input shape     : {x.shape}"
    )

    return x


# ============================================================
# Train / Validation Split
# ============================================================

def split_dataset(x):

    x_train, x_validation = train_test_split(
        x,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE,
    )

    print(
        f"Training samples   : {len(x_train)}"
    )

    print(
        f"Validation samples : {len(x_validation)}"
    )

    return (
        x_train,
        x_validation,
    )


# ============================================================
# TensorFlow Dataset
# ============================================================

def create_tf_dataset(
    x,
    shuffle=False,
):

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            x,
            x,
        )
    )

    if shuffle:

        dataset = dataset.shuffle(
            buffer_size=len(x),
            seed=RANDOM_STATE,
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# Build Model
# ============================================================

def build_model(
    input_shape,
):

    print("=" * 60)
    print("Building AutoEncoder")
    print("=" * 60)

    builder = AutoEncoder(
        input_shape=input_shape
    )

    model = builder.build()

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="mse",

        metrics=[
            tf.keras.metrics.MeanSquaredError(
                name="mse"
            )
        ],

    )

    model.summary()

    return model


# ============================================================
# Callbacks
# ============================================================

def create_callbacks():

    early_stopping = tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=PATIENCE,

        restore_best_weights=True,

        verbose=1,

    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(

        filepath=str(BEST_MODEL_PATH),

        monitor="val_loss",

        save_best_only=True,

        save_weights_only=False,

        verbose=1,

    )

    return [
        early_stopping,
        checkpoint,
    ]


# ============================================================
# Training
# ============================================================

def train(
    model,
    train_dataset,
    validation_dataset,
):

    print("=" * 60)
    print("AutoEncoder Training Started")
    print("=" * 60)

    history = model.fit(

        train_dataset,

        validation_data=validation_dataset,

        epochs=EPOCHS,

        callbacks=create_callbacks(),

        verbose=1,

    )

    return history


# ============================================================
# Save Training History
# ============================================================

def save_history(history):

    history_dataframe = pd.DataFrame(
        history.history
    )

    history_dataframe.insert(
        0,
        "epoch",
        range(
            1,
            len(history_dataframe) + 1,
        ),
    )

    history_dataframe.to_csv(
        HISTORY_PATH,
        index=False,
    )

    print(
        f"Training history saved : "
        f"{HISTORY_PATH}"
    )


# ============================================================
# Save Final Model
# ============================================================

def save_model(model):

    model.save(
        MODEL_PATH
    )

    print(
        f"Final model saved : "
        f"{MODEL_PATH}"
    )


# ============================================================
# Training Summary
# ============================================================

def print_summary(
    history,
    model,
):

    train_loss = history.history[
        "loss"
    ]

    validation_loss = history.history[
        "val_loss"
    ]

    best_epoch = (
        int(
            np.argmin(
                validation_loss
            )
        )
        + 1
    )

    best_validation_loss = min(
        validation_loss
    )

    print()
    print("=" * 60)
    print("Training Summary")
    print("=" * 60)

    print(
        f"Epochs completed       : "
        f"{len(train_loss)}"
    )

    print(
        f"Best epoch             : "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss  : "
        f"{best_validation_loss:.8f}"
    )

    print(
        f"Final training loss   : "
        f"{train_loss[-1]:.8f}"
    )

    print(
        f"Final validation loss : "
        f"{validation_loss[-1]:.8f}"
    )

    print(
        f"Model parameters      : "
        f"{model.count_params():,}"
    )

    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 60)
    print(" WAAM AutoEncoder Training ")
    print("=" * 60)
    print()

    # -----------------------------------------
    # 1. Prepare directories
    # -----------------------------------------

    prepare_directories()

    # -----------------------------------------
    # 2. Load normal dataset
    # -----------------------------------------

    x = load_dataset()

    # -----------------------------------------
    # 3. Validate input dimensions
    # -----------------------------------------

    if x.ndim != 4:

        raise ValueError(
            "AutoEncoder 입력 데이터는 "
            "(N, Height, Width, Channel) "
            "형태여야 합니다."
        )

    input_shape = x.shape[1:]

    print(
        f"AutoEncoder input shape : "
        f"{input_shape}"
    )

    # -----------------------------------------
    # 4. Train / Validation split
    # -----------------------------------------

    x_train, x_validation = split_dataset(
        x
    )

    # -----------------------------------------
    # 5. TensorFlow Dataset
    # -----------------------------------------

    train_dataset = create_tf_dataset(
        x_train,
        shuffle=True,
    )

    validation_dataset = create_tf_dataset(
        x_validation,
        shuffle=False,
    )

    # -----------------------------------------
    # 6. Build model
    # -----------------------------------------

    model = build_model(
        input_shape
    )

    # -----------------------------------------
    # 7. Train
    # -----------------------------------------

    history = train(

        model,

        train_dataset,

        validation_dataset,

    )

    # -----------------------------------------
    # 8. Save history
    # -----------------------------------------

    save_history(
        history
    )

    # -----------------------------------------
    # 9. Save final model
    # -----------------------------------------

    save_model(
        model
    )

    # -----------------------------------------
    # 10. Print summary
    # -----------------------------------------

    print_summary(
        history,
        model,
    )

    print()
    print(
        "AutoEncoder training completed."
    )
    print()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()