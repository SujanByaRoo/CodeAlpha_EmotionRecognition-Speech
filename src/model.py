"""
Phase 5: CNN architecture for speech emotion recognition.
Input: MFCC "images" of shape (n_mfcc, time_steps, 1)
Output: probability distribution over emotion classes
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, Flatten, Dense
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

        Flatten(),
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
    # Quick sanity check: build the model and print its summary
    model = build_cnn(input_shape=(40, 174, 1), num_classes=8)
    model.summary()