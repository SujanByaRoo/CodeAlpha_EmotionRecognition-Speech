"""
Phase 3: Extract MFCC features from RAVDESS audio files.
Converts each .wav into a fixed-size MFCC array, saves as a numpy dataset.
"""

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

METADATA_PATH = "data/processed/metadata.csv"
OUTPUT_X_PATH = "data/processed/X_mfcc.npy"
OUTPUT_Y_PATH = "data/processed/y_labels.npy"

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_PAD_LEN = 174  # ~4 seconds at default hop_length, covers RAVDESS clip lengths


def extract_mfcc(file_path, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    """Load an audio file and extract a fixed-size MFCC feature array."""
    audio, sr = librosa.load(file_path, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Pad or truncate along the time axis so every sample has the same shape
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc


def build_feature_dataset():
    df = pd.read_csv(METADATA_PATH)

    features = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting MFCCs"):
        try:
            mfcc = extract_mfcc(row["path"])
            features.append(mfcc)
            labels.append(row["emotion"])
        except Exception as e:
            print(f"Skipping {row['path']} due to error: {e}")

    X = np.array(features)
    y = np.array(labels)

    return X, y


if __name__ == "__main__":
    X, y = build_feature_dataset()

    print(f"\nFeature array shape: {X.shape}")   # (num_samples, n_mfcc, max_pad_len)
    print(f"Labels array shape: {y.shape}")

    np.save(OUTPUT_X_PATH, X)
    np.save(OUTPUT_Y_PATH, y)
    print(f"\nSaved features to {OUTPUT_X_PATH}")
    print(f"Saved labels to {OUTPUT_Y_PATH}")