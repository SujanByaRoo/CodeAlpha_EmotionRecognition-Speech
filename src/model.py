"""
Phase 5 (fixed): CNN architecture for speech emotion recognition.
Input: MFCC "images" of shape (n_mfcc, time_steps, 1)
Output: probability distribution over emotion classes

Uses GlobalAveragePooling2D instead of Flatten to drastically cut the
parameter count in the dense layers -- Flatten was producing a 1.7M-param
Dense layer for only ~900 training samples, which caused unstable training.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, GlobalAveragePooling2D, Dense
)


def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_cnn(input_shape=(40, 174, 1), num_classes=8)
    model.summary()