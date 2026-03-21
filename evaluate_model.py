import face_recognition
import cv2
import numpy as np
import os
import csv

# --- Configuration ---
KNOWN_FACES_DIR = "known_faces"
TEST_FACES_DIR = "test_dataset"
GROUND_TRUTH_FILE = "ground_truth.csv"
DISTANCE_THRESHOLD = 0.55  # The parameter you will tune!

# --- Step 1: Load Known Faces ---
# --- Updated Step 1: Load Known Faces with Debugging ---
print("Loading known faces...")
known_encodings = []
known_names = []

if not os.path.exists(KNOWN_FACES_DIR):
    print(f"Error: Folder '{KNOWN_FACES_DIR}' does not exist!")
    exit()

for file in os.listdir(KNOWN_FACES_DIR):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(KNOWN_FACES_DIR, file)
        img = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(file)[0].upper())
            print(f" Successfully loaded: {file}")
        else:
            print(f" [Warning] Could not find a face in: {file}")

if len(known_encodings) == 0:
    print(
        "Critical Error: No known faces were loaded. Check your 'known_faces' folder!"
    )
    exit()

# --- Step 2: Load Ground Truth ---
ground_truth = {}
try:
    with open(GROUND_TRUTH_FILE, mode="r") as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip header
        # Change rows[0] logic to handle the path
        for rows in reader:
            if len(rows) == 2:
                # Use the full path as the key in our dictionary
                ground_truth[rows[0].strip()] = rows[1].strip().upper()
except FileNotFoundError:
    print(f"Error: {GROUND_TRUTH_FILE} not found. Please create it.")
    exit()

# --- Step 3: Initialize Metrics ---
TP = 0  # True Positive: Correctly identified a known student
TN = 0  # True Negative: Correctly identified an unknown person
FP = 0  # False Positive: Identified a stranger as a student, or guessed the WRONG student
FN = 0  # False Negative: Failed to recognize a student who was there

print("\nStarting Evaluation...")
print("-" * 30)

# --- Step 4: Run Evaluation on Test Dataset ---
for root, dirs, files in os.walk(TEST_FACES_DIR):
    for filename in files:
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        folder_name = os.path.basename(root)
        rel_path = os.path.join(folder_name, filename)

        if rel_path not in ground_truth:
            print(f"[Skip] {rel_path} not in ground truth CSV.")
            continue

        actual_name = ground_truth[rel_path]
        img_path = os.path.join(root, filename)

        # --- EVERYTHING BELOW MUST BE INDENTED INSIDE THE LOOP ---

        # Load test image
        img = face_recognition.load_image_file(img_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        prediction = "UNKNOWN"

        if len(face_encodings) > 0:
            encoding = face_encodings[0]
            distances = face_recognition.face_distance(known_encodings, encoding)
            min_distance = np.min(distances)

            if min_distance < DISTANCE_THRESHOLD:
                prediction = known_names[np.argmin(distances)]

        # --- Step 5: Compare Prediction vs Actual (Also inside loop) ---
        if actual_name != "UNKNOWN":
            if prediction == actual_name:
                TP += 1
            elif prediction == "UNKNOWN":
                FN += 1
            else:
                FP += 1
        else:
            if prediction == "UNKNOWN":
                TN += 1
            else:
                FP += 1
# --- Step 6: Calculate and Print Metrics ---
total_images = TP + TN + FP + FN

# Protect against division by zero
accuracy = (TP + TN) / total_images if total_images > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = (
    2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
)

print("\n" + "=" * 30)
print(" FINAL ML METRICS ")
print("=" * 30)
print(f"Total Test Images Analyzed: {total_images}")
print(f"Distance Threshold Used : {DISTANCE_THRESHOLD}")
print("-" * 30)
print(f"True Positives (TP) : {TP}")
print(f"True Negatives (TN) : {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print("-" * 30)
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1_score * 100:.2f}%")
print("=" * 30)
