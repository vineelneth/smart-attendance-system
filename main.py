"""Real-time webcam-based facial recognition attendance system."""

import csv
import os
import sys
from datetime import datetime
from typing import Optional

import cv2
import face_recognition
import numpy as np

from config import (
    DISTANCE_THRESHOLD,
    FACE_AREA_SCALE,
    FACE_AREA_WEIGHT,
    FRAME_DELAY_MS,
    KNOWN_FACES_DIR,
    SHARPNESS_WEIGHT,
    STUDENTS_LIST_FILE,
    TOTAL_FRAMES,
    WARMUP_FRAMES,
)

_RESULT_WINDOW = "Smart Attendance — Result  (r=Retake | n=Next | q=Quit)"


def load_known_faces(directory: str) -> tuple[list, list]:
    """Load face encodings and names from all images in *directory*.

    Returns:
        (encodings, names) — parallel lists of face encodings and student names.
    """
    if not os.path.isdir(directory):
        print(
            f"Error: '{directory}/' not found.\n"
            "Create it and add one reference photo per student named 'StudentName.jpg'.\n"
            "See README for details."
        )
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
        else:
            print(f"[Warning] No face detected in '{file}' — skipping.")

    if not encodings:
        print(f"Error: No valid face images found in '{directory}/'.")
        sys.exit(1)

    print(f"Loaded {len(encodings)} known face(s).")
    return encodings, names


def load_student_list(filepath: str) -> list[str]:
    """Load enrolled student names from a plain-text file (one name per line)."""
    if not os.path.isfile(filepath):
        print(
            f"Error: '{filepath}' not found.\n"
            "Create it with one student name per line (uppercase)."
        )
        sys.exit(1)
    with open(filepath) as f:
        students = [line.strip().upper() for line in f if line.strip()]
    print(f"Loaded {len(students)} enrolled student(s).")
    return students


def measure_sharpness(img: np.ndarray) -> float:
    """Return the Laplacian variance of *img* as a sharpness proxy."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frame_quality_score(frame: np.ndarray) -> tuple[float, Optional[list]]:
    """Score *frame* by a weighted combination of sharpness and detected face area.

    Returns:
        (score, face_locations) — score is 0.0 and locations is None if no face found.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)
    if not faces:
        return 0.0, None
    sharpness = measure_sharpness(frame)
    avg_face_area = float(np.mean([(r - l) * (b - t) for (t, r, b, l) in faces]))
    score = SHARPNESS_WEIGHT * sharpness + FACE_AREA_WEIGHT * avg_face_area / FACE_AREA_SCALE
    return score, faces


def capture_best_frame(cap: cv2.VideoCapture) -> tuple[Optional[np.ndarray], Optional[list]]:
    """Capture TOTAL_FRAMES frames and return the sharpest one containing faces.

    The first WARMUP_FRAMES are discarded so the camera's auto-exposure can settle.
    Returns (None, None) if 'q' is pressed or no usable frame is found.
    """
    frames: list[np.ndarray] = []
    scores: list[float] = []
    face_sets: list[Optional[list]] = []

    for i in range(TOTAL_FRAMES):
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(
            frame, f"Capturing {i + 1}/{TOTAL_FRAMES}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        cv2.imshow("Smart Attendance — Capturing", frame)
        if cv2.waitKey(FRAME_DELAY_MS) & 0xFF == ord("q"):
            return None, None

        if i >= WARMUP_FRAMES:
            score, locs = frame_quality_score(frame)
            frames.append(frame)
            scores.append(score)
            face_sets.append(locs)

    if not frames:
        print("[Warning] No frames captured after warmup period.")
        return None, None

    best_idx = int(np.argmax(scores))
    best_frame = frames[best_idx]
    best_faces = face_sets[best_idx]

    if not best_faces:
        print("No clear faces detected. Try again.")
        return None, None

    print(f"Best frame: #{best_idx + WARMUP_FRAMES + 1}  |  {len(best_faces)} face(s) detected.")
    return best_frame, best_faces


def recognize_faces(
    frame: np.ndarray,
    face_locations: list,
    known_encodings: list,
    known_names: list,
    present_students: set,
) -> None:
    """Identify faces in *frame*, annotate it, and add matches to *present_students*."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for encoding, loc in zip(encodings, face_locations):
        distances = face_recognition.face_distance(known_encodings, encoding)
        if len(distances) == 0:
            continue

        min_dist = float(np.min(distances))
        y1, x2, y2, x1 = loc

        if min_dist < DISTANCE_THRESHOLD:
            name = known_names[int(np.argmin(distances))]
            present_students.add(name)
            color = (0, 255, 0)
            label = name
        else:
            color = (0, 0, 255)
            label = "Unknown"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1 + 6, y2 + 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)

    cv2.imshow(_RESULT_WINDOW, frame)


def save_attendance(all_students: list[str], present_students: set) -> str:
    """Write a timestamped attendance CSV and return its filename."""
    now = datetime.now()
    filename = f"Attendance_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    timestamp = now.strftime("%H:%M:%S")
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Student Name", "Status", "Timestamp"])
        for name in all_students:
            writer.writerow([name, "Present" if name in present_students else "Absent", timestamp])
    return filename


def print_summary(all_students: list[str], present_students: set, filename: str) -> None:
    """Print a human-readable attendance summary."""
    absent = sorted(s for s in all_students if s not in present_students)
    print("\n--- Attendance Summary ---")
    print(f"Present ({len(present_students)}): {', '.join(sorted(present_students)) or 'None'}")
    print(f"Absent  ({len(absent)}):  {', '.join(absent) or 'None'}")
    print(f"\nSaved: {filename}")


def main() -> None:
    print("Loading known faces...")
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)
    all_students = load_student_list(STUDENTS_LIST_FILE)
    present_students: set[str] = set()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        sys.exit(1)

    print("\nCamera ready.  Controls:  r = Retake  |  n = Next group  |  q = Quit\n")

    try:
        while True:
            best_frame, best_faces = capture_best_frame(cap)
            if best_frame is None:
                break

            recognize_faces(best_frame, best_faces, known_encodings, known_names, present_students)

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("r"):
                    print("Retaking...")
                    cv2.destroyWindow(_RESULT_WINDOW)
                    break
                elif key == ord("n"):
                    print("Next group...")
                    cv2.destroyWindow(_RESULT_WINDOW)
                    break
                elif key == ord("q"):
                    print("Saving attendance...")
                    cv2.destroyWindow(_RESULT_WINDOW)
                    return
    finally:
        cap.release()
        cv2.destroyAllWindows()
        filename = save_attendance(all_students, present_students)
        print_summary(all_students, present_students, filename)


if __name__ == "__main__":
    main()
