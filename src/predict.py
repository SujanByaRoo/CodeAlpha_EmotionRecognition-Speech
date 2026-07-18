"""
Phase 8: Predict emotion from a new, unseen audio file.
Usage: python src/predict.py path/to/audio.wav
"""

import sys
import numpy as np
import librosa
from tensorflow.keras.models import load_model

MODEL_PATH = "models/emotion_cnn.h5"
MEAN_PATH = "data/processed/mfcc_mean.npy"
STD_PATH = "data/processed/mfcc_std.npy"
CLASSES_PATH = "data/processed/label_classes.npy"

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_PAD_LEN = 174


def extract_mfcc(file_path, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc


def predict_emotion(file_path):
    # Load model + saved normalization stats (must match training exactly)
    model = load_model(MODEL_PATH)
    mean = np.load(MEAN_PATH)
    std = np.load(STD_PATH)
    class_names = np.load(CLASSES_PATH, allow_pickle=True)

    # Extract and normalize features the same way as during training
    mfcc = extract_mfcc(file_path)
    mfcc = (mfcc - mean.squeeze()[:, np.newaxis]) / std.squeeze()[:, np.newaxis]
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # add batch + channel dims

    # Predict
    probs = model.predict(mfcc, verbose=0)[0]
    predicted_idx = np.argmax(probs)
    predicted_emotion = class_names[predicted_idx]

    print(f"\nFile: {file_path}")
    print(f"Predicted emotion: {predicted_emotion}\n")
    print("Confidence per class:")
    for name, prob in sorted(zip(class_names, probs), key=lambda x: -x[1]):
        print(f"  {name:12s}: {prob*100:5.1f}%")

    return predicted_emotion


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/audio.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    predict_emotion(audio_path)