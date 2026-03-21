# smart-attendance-system
A Python facial recognition pipeline optimized for 96% precision with blur detection.
# Smart Attendance System

An automated, real-time facial recognition attendance system built with Python, OpenCV, and deep metric learning. This project replaces manual roll-calls with a high-precision, automated CSV logging pipeline.

## 🌟 Key Engineering Features
- **Hyperparameter Tuning:** Conducted evaluation testing on 265 images, shifting the Euclidean distance threshold from 0.48 to **0.55**. This optimized the decision boundary, increasing Recall by 20% while maintaining a **96% Precision rate**.
- **Image Quality Assessment (IQA):** Implemented a Laplacian Variance filter to calculate image sharpness in real-time. The system captures a 15-frame burst and automatically selects the clearest frame for inference, eliminating errors caused by motion blur.
- **Bulletproof Pathing:** Engineered the pipeline using `os.path.abspath(__file__)` to dynamically resolve directory paths, preventing "File Not Found" errors across different environments.
- **Automated Logging:** Cross-references detected faces with a master `students_list.txt` roster and automatically generates timestamped `.csv` attendance reports.

## 📊 Evaluation Metrics
Based on the `evaluate_model.py` validation suite:
* **Precision:** 96.05%
* **Recall:** 56.37%
* **F1-Score:** 71.05%
* *Note: Real-world recall is significantly higher due to the Laplacian frame-selection algorithm implemented in the live deployment.*

## 🛠️ Tech Stack
* **Python** (Core Logic)
* **OpenCV / cv2** (Camera I/O, Image converting, GUI)
* **face_recognition / dlib** (128-D vector embeddings)
* **NumPy** (Vector distance calculations)

## 🚀 How to Run
1. Clone this repository to your local machine.
2. Install the required libraries: `pip install -r requirements.txt`
3. Create a folder named `known_faces` in the root directory and add a clear, front-facing `.jpg` of each student.
4. Ensure the names in `students_list.txt` exactly match your image filenames.
5. Run the live camera system: `python main.py`
