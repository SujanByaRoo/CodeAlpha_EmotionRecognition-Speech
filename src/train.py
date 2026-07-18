"""
Phase 6 (fixed): Train the CNN on MFCC features to classify emotions.
Uses a lower learning rate + ReduceLROnPlateau to fix the training
instability seen with the default Adam learning rate (loss was stuck
near ln(8), i.e. random-guessing level, and val_loss was spiking).
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from model import build_cnn

DATA_DIR = "data/processed"
MODEL_SAVE_PATH = "models/emotion_cnn.h5"
HISTORY_PLOT_PATH = "models/training_history.png"

EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 1e-4


def load_data():
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    X_val = np.load(f"{DATA_DIR}/X_val.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    y_val = np.load(f"{DATA_DIR}/y_val.npy")
    return X_train, X_val, y_train, y_val


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH)
    print(f"\nSaved training curves to {HISTORY_PLOT_PATH}")


def train():
    X_train, X_val, y_train, y_val = load_data()

    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]

    model = build_cnn(input_shape, num_classes)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    plot_history(history)
    print(f"\nBest model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()