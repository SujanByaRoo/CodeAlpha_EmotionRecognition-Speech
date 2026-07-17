"""
Phase 2: Explore RAVDESS dataset structure.
Parses filenames to build a dataframe of [file_path, emotion].
"""

import os
import pandas as pd

DATA_DIR = "data/raw"

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def build_dataframe(data_dir=DATA_DIR):
    file_paths = []
    emotions = []

    for actor_folder in os.listdir(data_dir):
        actor_path = os.path.join(data_dir, actor_folder)
        if not os.path.isdir(actor_path):
            continue

        for file_name in os.listdir(actor_path):
            if not file_name.endswith(".wav"):
                continue

            parts = file_name.split("-")
            emotion_code = parts[2]  # 3rd number encodes emotion
            emotion = EMOTION_MAP.get(emotion_code)

            if emotion is None:
                continue

            file_paths.append(os.path.join(actor_path, file_name))
            emotions.append(emotion)

    df = pd.DataFrame({"path": file_paths, "emotion": emotions})
    return df


if __name__ == "__main__":
    df = build_dataframe()
    print(f"Total files found: {len(df)}")
    print("\nClass distribution:")
    print(df["emotion"].value_counts())

    # Save for reuse in later phases
    df.to_csv("data/processed/metadata.csv", index=False)
    print("\nSaved metadata to data/processed/metadata.csv")