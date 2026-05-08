"""Central configuration for the Smart Attendance System."""

# --- Paths ---
KNOWN_FACES_DIR = "known_faces"
TEST_FACES_DIR = "test_dataset"
STUDENTS_LIST_FILE = "students_list.txt"
GROUND_TRUTH_FILE = "data.csv"

# --- Face Recognition ---
# Lower = stricter matching (fewer false positives, more false negatives).
# Tuned to 0.55 via evaluate_model.py on the bundled test dataset.
DISTANCE_THRESHOLD = 0.55

# --- Frame Capture ---
TOTAL_FRAMES = 15
# Skip the first N frames so the camera's auto-exposure can stabilize.
WARMUP_FRAMES = 4
FRAME_DELAY_MS = 150

# --- Quality Score Weights ---
SHARPNESS_WEIGHT = 0.7
FACE_AREA_WEIGHT = 0.3
FACE_AREA_SCALE = 1000  # Normalization factor for face-area component
