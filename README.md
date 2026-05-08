# Smart Attendance System

A real-time facial recognition attendance system built with Python and OpenCV. Point a webcam at a group, and the system automatically identifies registered students, marks them present, and saves a timestamped CSV report.

---

## Features

- **Real-time face detection** via webcam with multi-face support
- **Smart frame selection** — captures 15 frames and picks the sharpest one with the largest faces (skips the first 4 frames for camera auto-exposure to settle)
- **Configurable recognition threshold** — tune sensitivity without touching the model
- **Offline evaluation suite** — measure accuracy, precision, recall, and F1-score against a labeled test dataset
- **Timestamped CSV reports** — one file per session with Present/Absent status for every enrolled student

---

## How It Works

```
Enroll students          Run system              Output
─────────────────        ──────────────          ────────────────────────────
known_faces/             1. Capture 15 frames    Attendance_2026-03-21.csv
  john_doe.jpg    ──▶    2. Select sharpest  ──▶  John Doe, Present, 09:07:12
  jane_smith.jpg         3. Encode faces          Jane Smith, Absent, 09:07:12
students_list.txt        4. Match encodings
                            (Euclidean distance)
```

Each student photo is converted to a 128-dimensional face embedding using dlib's ResNet model. At recognition time, each detected face is compared against all known embeddings; a match is accepted if the Euclidean distance falls below the configured threshold (default: **0.55**).

---

## Project Structure

```
smart_attendance_system/
├── main.py               # Webcam-based attendance marking
├── evaluate_model.py     # Offline model evaluation
├── config.py             # All tunable parameters (threshold, paths, weights)
├── students_list.txt     # Enrolled student names (one per line, uppercase)
├── data.csv              # Ground-truth labels for evaluation
├── requirements.txt
├── known_faces/          # ← Add one reference photo per student here
│   └── Student_Name.jpg
└── test_dataset/         # ← Add labeled test images here (for evaluation)
    └── PersonName/
        ├── image_1.jpg
        └── ...
```

---

## Setup

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.8+ | |
| Webcam | Required for `main.py` only |
| **CMake** | Windows only — [cmake.org/download](https://cmake.org/download/) |
| **VS Build Tools** | Windows only — needed to compile dlib |

> **Windows users:** Install CMake and [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (select "Desktop development with C++") **before** running `pip install`.

### Installation

```bash
# 1. Clone
git clone https://github.com/yourusername/smart-attendance-system.git
cd smart-attendance-system

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Enrolling Students

1. Add one clear, front-facing photo per student to `known_faces/`:
   ```
   known_faces/
     John_Doe.jpg
     Jane_Smith.jpg
   ```
   The filename (without extension) becomes the student's display name.

2. Add the same names (uppercase) to `students_list.txt`:
   ```
   JOHN DOE
   JANE SMITH
   ```
   Students on this list but not recognized in a session are marked **Absent**.

> Each photo must contain exactly one face. Images with no detectable face are skipped with a warning.

---

## Usage

### Mark Attendance

```bash
python main.py
```

The system captures frames from your webcam, selects the best one, and displays recognition results.

| Key | Action |
|-----|--------|
| `r` | Retake the current group |
| `n` | Move to the next group of students |
| `q` | Save attendance and exit |

**Output** — a timestamped CSV is created automatically:

```
Student Name,Status,Timestamp
JOHN DOE,Present,09:07:12
JANE SMITH,Absent,09:07:12
```

### Evaluate the Model

Requires a `test_dataset/` directory and a `data.csv` ground-truth file.

```bash
python evaluate_model.py
```

**Ground-truth CSV format** (`data.csv`):

```
filepath,actual_name
PersonName/image_1.jpg,PERSON NAME
PersonName/image_2.jpg,PERSON NAME
unknown_person/image.jpg,UNKNOWN
```

**Sample output:**

```
====================================
        FINAL ML METRICS
====================================
Total Test Images  : 266
Distance Threshold : 0.55
------------------------------------
True Positives  (TP) : 241
True Negatives  (TN) :  18
False Positives (FP) :   4
False Negatives (FN) :   3
------------------------------------
Accuracy  : 97.37%
Precision : 98.37%
Recall    : 98.77%
F1-Score  : 98.57%
====================================
```

### Tuning the Threshold

Edit `config.py`:

```python
DISTANCE_THRESHOLD = 0.55  # default
```

| Value | Effect |
|-------|--------|
| Lower (e.g. `0.45`) | Stricter — fewer false positives, more false negatives |
| Higher (e.g. `0.65`) | More lenient — fewer false negatives, more false positives |

Run `evaluate_model.py` after each change to measure the impact.

---

## Configuration Reference

All settings live in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DISTANCE_THRESHOLD` | `0.55` | Face match sensitivity |
| `TOTAL_FRAMES` | `15` | Frames captured per recognition attempt |
| `WARMUP_FRAMES` | `4` | Frames discarded for camera auto-exposure |
| `FRAME_DELAY_MS` | `150` | Delay between captured frames (ms) |
| `SHARPNESS_WEIGHT` | `0.7` | Weight of sharpness in frame quality score |
| `FACE_AREA_WEIGHT` | `0.3` | Weight of face area in frame quality score |

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| [face_recognition](https://github.com/ageitgey/face_recognition) | Face encoding and matching (powered by dlib) |
| [OpenCV](https://opencv.org/) | Webcam capture, image processing, UI overlays |
| [NumPy](https://numpy.org/) | Vectorized distance calculations |

---

## License

[MIT](LICENSE)
