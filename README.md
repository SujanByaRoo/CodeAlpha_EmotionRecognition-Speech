# Emotion Recognition from Speech 🎤



The goal is simple — give it an audio clip of someone talking, and it tries to guess what emotion they're feeling (happy, sad, angry, etc.) just from their voice.

## What I did

1. Used the **RAVDESS** dataset — a bunch of actors saying the same lines in different emotions.
2. Extracted **MFCC features** from each audio file using `librosa` (basically turns sound into numbers a model can learn from).
3. Built a **CNN** (Convolutional Neural Network) using TensorFlow/Keras and trained it to classify the MFCCs into 8 emotions.
4. Evaluated it using accuracy, precision, recall, F1-score, and a confusion matrix.
5. Wrote a script so you can throw in any new `.wav` file and get a prediction.

## Results

Got about **47-49% test accuracy** across 8 emotion classes.

Some emotions the model is really good at spotting — like **surprised** and **angry** (these have very distinct energy/pitch patterns in speech).

It struggles more with **happy** — it keeps confusing it with angry or calm. Turns out this is a known issue in speech emotion recognition — MFCCs are better at picking up how *intense* someone sounds rather than whether they sound *positive or negative*. So happy (high energy + positive) gets mixed up with angry (high energy + negative) pretty easily.

Check out `models/confusion_matrix.png` and `models/training_history.png` to see this visually.

## Project structure

```
├── data/
│   ├── raw/              # RAVDESS audio files
│   └── processed/        # extracted features, splits, etc
├── models/
│   ├── emotion_cnn.h5           # the trained model
│   ├── training_history.png     # accuracy/loss graphs
│   └── confusion_matrix.png     # how well it did per class
├── src/
│   ├── explore_data.py          # reads RAVDESS filenames, labels emotions
│   ├── feature_extraction.py    # turns audio into MFCCs
│   ├── prepare_dataset.py       # normalizes + splits data for training
│   ├── model.py                 # the CNN architecture
│   ├── train.py                 # trains the model
│   ├── evaluate.py              # tests the model, makes confusion matrix
│   └── predict.py               # predict emotion on a new audio file
└── requirements.txt
```

## How to run it

**1. Set up the environment**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Get the dataset**
Download `Audio_Speech_Actors_01-24.zip` from [RAVDESS on Zenodo](https://zenodo.org/record/1188976), extract it into `data/raw/` so you get `data/raw/Actor_01/`, `data/raw/Actor_02/`, etc.

**3. Run everything in order**
```bash
python src/explore_data.py
python src/feature_extraction.py
python src/prepare_dataset.py
python src/train.py
python src/evaluate.py
```

**4. Try it on your own audio**
```bash
python src/predict.py path/to/your/audio.wav
```

## Tools used

- Python
- TensorFlow / Keras (CNN)
- librosa (for MFCC extraction)
- scikit-learn (splitting data, metrics)
- matplotlib / seaborn (graphs)

---

