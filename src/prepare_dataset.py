"""
Phase 4 (fixed): Prepare dataset for CNN training.
Loads MFCC features + labels, encodes labels, splits into train/val/test,
normalizes PER MFCC COEFFICIENT using train-set statistics only (fixes
unstable training caused by mismatched feature scales), reshapes for CNN.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

X_PATH = "data/processed/X_mfcc.npy"
Y_PATH = "data/processed/y_labels.npy"

OUT_DIR = "data/processed"


def load_and_prepare():
    X = np.load(X_PATH)   # shape: (samples, n_mfcc, time_steps)
    y = np.load(Y_PATH)

    # Encode string labels ("happy", "sad", ...) into integers, then one-hot
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_int)

    print("Classes (in order of encoding):", list(label_encoder.classes_))

    # Split RAW (unnormalized) features first, so we compute normalization
    # stats only from the training set (avoids leaking test/val info)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )

    # Per-coefficient normalization: compute mean/std for each of the 40
    # MFCC coefficients across the training set (axis 0=samples, 2=time)
    mean = X_train.mean(axis=(0, 2), keepdims=True)  # shape (1, 40, 1)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    # Add channel dimension: (samples, n_mfcc, time_steps) -> (..., 1)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print(f"\nTrain shape: {X_train.shape}, {y_train.shape}")
    print(f"Val shape:   {X_val.shape}, {y_val.shape}")
    print(f"Test shape:  {X_test.shape}, {y_test.shape}")

    np.save(f"{OUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUT_DIR}/y_test.npy", y_test)
    np.save(f"{OUT_DIR}/label_classes.npy", label_encoder.classes_)

    # Save normalization stats too — needed later for inference on new audio
    np.save(f"{OUT_DIR}/mfcc_mean.npy", mean)
    np.save(f"{OUT_DIR}/mfcc_std.npy", std)

    print("\nSaved train/val/test splits and normalization stats to data/processed/")


if __name__ == "__main__":
    load_and_prepare()