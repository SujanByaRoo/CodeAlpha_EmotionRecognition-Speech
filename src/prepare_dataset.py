"""
Phase 4: Prepare dataset for CNN training.
Loads MFCC features + labels, encodes labels, splits into train/val/test,
reshapes features to (samples, n_mfcc, time_steps, 1) for CNN input.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

X_PATH = "data/processed/X_mfcc.npy"
Y_PATH = "data/processed/y_labels.npy"

OUT_DIR = "data/processed"


def load_and_prepare():
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    # Normalize features (helps CNN training converge faster)
    X = (X - np.mean(X)) / np.std(X)

    # Add channel dimension: (samples, n_mfcc, time_steps) -> (samples, n_mfcc, time_steps, 1)
    X = X[..., np.newaxis]

    # Encode string labels ("happy", "sad", ...) into integers, then one-hot
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_int)

    print("Classes (in order of encoding):", list(label_encoder.classes_))

    # First split: train+val vs test (80/20)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
    )

    # Second split: train vs val (80/20 of the remaining 80%, i.e. 64/16/20 overall)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )

    print(f"\nTrain shape: {X_train.shape}, {y_train.shape}")
    print(f"Val shape:   {X_val.shape}, {y_val.shape}")
    print(f"Test shape:  {X_test.shape}, {y_test.shape}")

    # Save everything for the training phase
    np.save(f"{OUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUT_DIR}/y_test.npy", y_test)
    np.save(f"{OUT_DIR}/label_classes.npy", label_encoder.classes_)

    print("\nSaved train/val/test splits to data/processed/")


if __name__ == "__main__":
    load_and_prepare()