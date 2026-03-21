import face_recognition
import cv2
import numpy as np
import os
import csv
from datetime import datetime

# --- Configuration ---
KNOWN_FACES_DIR = "known_faces"
STUDENTS_LIST_FILE = "students_list.txt"
DISTANCE_THRESHOLD = 0.55  # Updated from 0.48 based on evaluation tuning

# ---------- STEP 1: Load known faces ----------
known_encodings = []
known_names = []

print("Loading known faces...")
for file in os.listdir(KNOWN_FACES_DIR):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(KNOWN_FACES_DIR, file)
        img = face_recognition.load_image_file(img_path)
        enc = face_recognition.face_encodings(img)
        if len(enc) > 0:
            known_encodings.append(enc[0])
            known_names.append(os.path.splitext(file)[0].upper())
        else:
            print(f"[Warning] No face found in {file}")

# ---------- STEP 2: Read student list ----------
try:
    with open(STUDENTS_LIST_FILE, "r") as f:
        all_students = [line.strip().upper() for line in f.readlines()]
except FileNotFoundError:
    print(f"Error: '{STUDENTS_LIST_FILE}' not found. Please create it.")
    exit()

present_students = set()


# ---------- Helper Functions ----------
def measure_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def frame_quality_score(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)
    if len(faces) == 0:
        return 0, None
    sharpness = measure_sharpness(frame)
    face_areas = [(r - l) * (b - t) for (t, r, b, l) in faces]
    avg_face_area = np.mean(face_areas)
    score = 0.7 * sharpness + 0.3 * avg_face_area / 1000
    return score, faces


def capture_best_frame(cap):
    """Capture frames 5 to 15 and pick the best (clearest) one."""
    frames, scores, face_sets = [], [], []
    total_frames = 15

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(
            frame,
            f"Capturing frame {i+1}/{total_frames}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Capturing Best Frame", frame)
        cv2.waitKey(150)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None, None

        # Store only frames 5 to 15
        if i >= 4:
            frames.append(frame)

    if not frames:
        print("No frames captured between 5 and 15.")
        return None, None

    # Evaluate sharpness and face clarity only for frames 5–15
    for f in frames:
        s, locs = frame_quality_score(f)
        scores.append(s)
        face_sets.append(locs)

    best_idx = np.argmax(scores)
    best_frame = frames[best_idx]
    best_faces = face_sets[best_idx]

    if not best_faces:
        print("No clear faces detected in frames 5–15.")
        return None, None

    print(f"Best frame selected (from frames 5–15) -> Frame #{best_idx + 5}")
    return best_frame, best_faces


def recognize_faces(best_frame, best_faces):
    """Recognize all known faces in selected frame."""
    rgb_best = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_best, best_faces)

    for face_encoding, face_loc in zip(face_encodings, best_faces):
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(distances) == 0:
            continue

        min_distance = np.min(distances)
        if min_distance < DISTANCE_THRESHOLD:
            name = known_names[np.argmin(distances)]
            present_students.add(name)
            y1, x2, y2, x1 = face_loc
            cv2.rectangle(best_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                best_frame,
                name,
                (x1 + 6, y2 + 25),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        else:
            y1, x2, y2, x1 = face_loc
            cv2.rectangle(best_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                best_frame,
                "Unknown",
                (x1 + 6, y2 + 25),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

    cv2.imshow("Result - Press 'r' to Retake, 'n' for Next, 'q' to Quit", best_frame)


# ---------- STEP 3: Initialize Camera ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

print(
    "\nCamera ready! Press keys to control:\n'r' = Retake, 'n' = Next group, 'q' = Quit\n"
)

# ---------- STEP 4: Main Loop ----------
goto_exit = False
while not goto_exit:
    best_frame, best_faces = capture_best_frame(cap)
    if best_frame is None:
        continue

    print(f"Best frame selected with {len(best_faces)} face(s).")
    recognize_faces(best_frame, best_faces)

    while True:
        key = cv2.waitKey(0) & 0xFF  # Wait for key input
        if key == ord("r"):  # Retake
            print("\nRetaking current group...")
            cv2.destroyWindow("Result - Press 'r' to Retake, 'n' for Next, 'q' to Quit")
            break
        elif key == ord("n"):  # Next
            print("\nMoving to next group...")
            cv2.destroyWindow("Result - Press 'r' to Retake, 'n' for Next, 'q' to Quit")
            break
        elif key == ord("q"):  # Quit
            print("\nExiting and saving attendance...")
            cap.release()
            cv2.destroyAllWindows()
            goto_exit = True
            break

# ---------- STEP 5: Save Attendance ----------
absent_students = [s for s in all_students if s not in present_students]
now = datetime.now()
filename = f"Attendance_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv"

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Student Name", "Status", "Timestamp"])
    for name in all_students:
        status = "Present" if name in present_students else "Absent"
        writer.writerow([name, status, now.strftime("%H:%M:%S")])

print("\nAttendance Summary:")
print("\nPresent Students:")
for s in sorted(present_students):
    print("  -", s)

print("\nAbsent Students:")
for s in sorted(absent_students):
    print("  -", s)

print(f"\nAttendance saved in: {filename}")
