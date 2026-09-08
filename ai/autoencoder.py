"""
ai/autoencoder.py

WAAM Convolutional AutoEncoder

Author : Yunseo Anna Jung

Real-Time Welding Sound
Anomaly Detection Model

Input :
    Mel Spectrogram

Output :
    Reconstructed Mel Spectrogram
"""

from __future__ import annotations

import tensorflow as tf

from tensorflow.keras import Model # type: ignore
from tensorflow.keras.layers import ( # type: ignore
    Input,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
)


class AutoEncoder:

    """
    Convolutional AutoEncoder Builder
    """

    def __init__(

        self,

        input_shape=(128, 64, 1),

    ):

        self.input_shape = input_shape

    # ======================================================

    def build_encoder(self):

        inputs = Input(shape=self.input_shape)

        x = Conv2D(
            16,
            3,
            activation="relu",
            padding="same",
        )(inputs)

        x = MaxPooling2D(
            2,
            padding="same",
        )(x)

        x = Conv2D(
            32,
            3,
            activation="relu",
            padding="same",
        )(x)

        x = MaxPooling2D(
            2,
            padding="same",
        )(x)

        x = Conv2D(
            64,
            3,
            activation="relu",
            padding="same",
        )(x)

        latent = MaxPooling2D(
            2,
            padding="same",
            name="latent_space",
        )(x)

        encoder = Model(

            inputs,

            latent,

            name="Encoder",

        )

        return encoder

    # ======================================================

    def build_decoder(self):

        latent_shape = (

            self.input_shape[0] // 8,

            self.input_shape[1] // 8,

            64,

        )

        inputs = Input(shape=latent_shape)

        x = Conv2DTranspose(

            64,

            3,

            strides=2,

            padding="same",

            activation="relu",

        )(inputs)

        x = Conv2DTranspose(

            32,

            3,

            strides=2,

            padding="same",

            activation="relu",

        )(x)

        x = Conv2DTranspose(

            16,

            3,

            strides=2,

            padding="same",

            activation="relu",

        )(x)

        outputs = Conv2D(

            1,

            3,

            activation="sigmoid",

            padding="same",

        )(x)

        decoder = Model(

            inputs,

            outputs,

            name="Decoder",

        )

        return decoder

    # ======================================================

    def build(self):

        encoder = self.build_encoder()

        decoder = self.build_decoder()

        inputs = Input(shape=self.input_shape)

        latent = encoder(inputs)

        outputs = decoder(latent)

        model = Model(

            inputs,

            outputs,

            name="WAAM_AutoEncoder",

        )

        return model