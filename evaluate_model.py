"""Offline evaluation of face recognition accuracy against a labeled test dataset."""

import csv
import os
import sys
from typing import Optional

import face_recognition
import numpy as np

from config import (
    DISTANCE_THRESHOLD,
    GROUND_TRUTH_FILE,
    KNOWN_FACES_DIR,
    TEST_FACES_DIR,
)


def load_known_faces(directory: str) -> tuple[list, list]:
    """Load face encodings and names from all images in *directory*.

    Returns:
        (encodings, names) — parallel lists.
    """
    if not os.path.isdir(directory):
        print(f"Error: '{directory}/' not found. See README for setup.")
        sys.exit(1)

    encodings: list = []
    names: list = []

    for file in os.listdir(directory):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = face_recognition.load_image_file(os.path.join(directory, file))
        enc = face_recognition.face_encodings(img)
        if enc:
            encodings.append(enc[0])
            names.append(os.path.splitext(file)[0].upper())
            print(f"  Loaded: {file}")
        else:
            print(f"  [Warning] No face in '{file}' — skipped.")

    if not encodings:
        print(f"Error: No valid face images in '{directory}/'.")
        sys.exit(1)

    return encodings, names


def load_ground_truth(filepath: str) -> dict[str, str]:
    """Load filepath → actual_name mapping from a two-column CSV file.

    Path separators are normalised to '/' for cross-platform consistency.
    """
    ground_truth: dict[str, str] = {}
    try:
        with open(filepath, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header row
            for row in reader:
                if len(row) == 2:
                    key = row[0].strip().replace("\\", "/")
                    ground_truth[key] = row[1].strip().upper()
    except FileNotFoundError:
        print(f"Error: Ground-truth file '{filepath}' not found.")
        sys.exit(1)
    return ground_truth


def predict(
    img_path: str,
    known_encodings: list,
    known_names: list,
    threshold: float,
) -> Optional[str]:
    """Return the predicted label for *img_path*, or None if no face is detected.

    Returns 'UNKNOWN' if the closest match exceeds *threshold*.
    """
    img = face_recognition.load_image_file(img_path)
    locations = face_recognition.face_locations(img)
    encodings = face_recognition.face_encodings(img, locations)

    if not encodings:
        return None

    distances = face_recognition.face_distance(known_encodings, encodings[0])
    min_dist = float(np.min(distances))

    if min_dist < threshold:
        return known_names[int(np.argmin(distances))]
    return "UNKNOWN"


def run_evaluation(
    test_dir: str,
    ground_truth: dict[str, str],
    known_encodings: list,
    known_names: list,
    threshold: float,
) -> dict[str, int]:
    """Walk *test_dir*, predict each image, and tally TP/TN/FP/FN counts."""
    if not os.path.isdir(test_dir):
        print(f"Error: Test dataset directory '{test_dir}/' not found.")
        sys.exit(1)

    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for root, _, files in os.walk(test_dir):
        for filename in files:
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            folder_name = os.path.basename(root)
            # Normalize separators so paths match ground-truth CSV keys on all platforms.
            rel_path = os.path.join(folder_name, filename).replace("\\", "/")

            if rel_path not in ground_truth:
                print(f"[Skip] {rel_path} — not in ground truth.")
                continue

            actual = ground_truth[rel_path]
            prediction = predict(os.path.join(root, filename), known_encodings, known_names, threshold)
            prediction = prediction or "UNKNOWN"

            if actual != "UNKNOWN":
                if prediction == actual:
                    counts["TP"] += 1
                elif prediction == "UNKNOWN":
                    counts["FN"] += 1
                else:
                    counts["FP"] += 1  # wrong student identified
            else:
                if prediction == "UNKNOWN":
                    counts["TN"] += 1
                else:
                    counts["FP"] += 1  # unknown person misidentified as a student

    return counts


def print_metrics(counts: dict[str, int], threshold: float) -> None:
    """Print a formatted evaluation report to stdout."""
    tp, tn, fp, fn = counts["TP"], counts["TN"], counts["FP"], counts["FN"]
    total = tp + tn + fp + fn

    accuracy  = (tp + tn) / total        if total > 0           else 0.0
    precision = tp / (tp + fp)           if (tp + fp) > 0       else 0.0
    recall    = tp / (tp + fn)           if (tp + fn) > 0       else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 36)
    print("        FINAL ML METRICS        ")
    print("=" * 36)
    print(f"Total Test Images  : {total}")
    print(f"Distance Threshold : {threshold}")
    print("-" * 36)
    print(f"True Positives  (TP) : {tp}")
    print(f"True Negatives  (TN) : {tn}")
    print(f"False Positives (FP) : {fp}")
    print(f"False Negatives (FN) : {fn}")
    print("-" * 36)
    print(f"Accuracy  : {accuracy  * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall    * 100:.2f}%")
    print(f"F1-Score  : {f1        * 100:.2f}%")
    print("=" * 36)


def main() -> None:
    print("Loading known faces...")
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)
    print(f"  → {len(known_encodings)} face(s) loaded.\n")

    ground_truth = load_ground_truth(GROUND_TRUTH_FILE)
    print(f"Ground truth: {len(ground_truth)} entries loaded from '{GROUND_TRUTH_FILE}'.\n")

    print("Running evaluation...")
    counts = run_evaluation(TEST_FACES_DIR, ground_truth, known_encodings, known_names, DISTANCE_THRESHOLD)
    print_metrics(counts, DISTANCE_THRESHOLD)


if __name__ == "__main__":
    main()
