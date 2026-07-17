"""
Phase 7: Evaluate the trained CNN on the held-out test set.
Reports accuracy, precision, recall, F1-score, and a confusion matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_DIR = "data/processed"
MODEL_PATH = "models/emotion_cnn.h5"
CONFUSION_MATRIX_PATH = "models/confusion_matrix.png"


def load_test_data():
    X_test = np.load(f"{DATA_DIR}/X_test.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")
    class_names = np.load(f"{DATA_DIR}/label_classes.npy", allow_pickle=True)
    return X_test, y_test, class_names


def evaluate():
    X_test, y_test, class_names = load_test_data()
    model = load_model(MODEL_PATH)

    # Predict
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}\n")

    # Precision, Recall, F1 per class
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Emotion Recognition")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    evaluate()