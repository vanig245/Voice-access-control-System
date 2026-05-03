# Voice Access Control System

A machine learning-based voice authentication system that grants or denies access based on speaker identity. Built with Python, Flask, and Librosa — trained on MFCC voice features using an SVM classifier.

---

## How It Works

1. Voice Recording — User speaks into the browser microphone for 3 seconds
2. Feature Extraction — Audio is converted to MFCC (Mel-Frequency Cepstral Coefficients) features using Librosa
3. Prediction — A trained SVM model identifies the speaker
4. Access Decision — If the speaker is in the authorized users list → Access Granted, else → Access Denied

---

## Project Structure

```
Voice-Access-Control-System/
│
├── voice_data/                  # Training voice samples
│   ├── user1/                   # 10 WAV files per user
│   │   ├── user1_1.wav
│   │   ├── user1_2.wav
│   │   └── ... (10 samples)
│   ├── user2/
│   ├── user3/
│   ├── user4/
│   └── user5/
│
├── voice-access-web/            # Flask web application
│   ├── templates/
│   │   └── index.html           # Frontend UI (record + display result)
│   ├── app.py                   # Flask backend + prediction logic
│   ├── voice_model.pkl          # Trained SVM model
│   ├── temp.wav                 # Temporary audio file (auto-generated)
│   └── temp.webm                # Temporary audio file (auto-generated)
│
├── main.py                      # Model training script
├── .venv/                       # Virtual environment
└── README.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.x |
| Web Framework | Flask |
| Audio Processing | Librosa |
| ML Model | Scikit-learn (SVM) |
| Feature Extraction | MFCC (13 coefficients) |
| Frontend | HTML, CSS, JavaScript |
| Audio Recording | Web MediaRecorder API |
| Audio Conversion | FFmpeg |

---

## Getting Started

### Prerequisites
Make sure you have these installed:
- Python 3.8+
- FFmpeg — [Download here](https://ffmpeg.org/download.html)
- pip

---

### 1. Clone the Repository
```bash
git clone https://github.com/vanig245/Voice-access-control-System.git
cd Voice-access-control-System
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install flask librosa scikit-learn numpy soundfile
```

### 4. Add Your Voice Data
Place WAV audio files in this structure:
```
voice_data/
  user1/
    user1_1.wav
    user1_2.wav
    ... (minimum 5-10 samples recommended)
  user2/
    user2_1.wav
    ...
```

### 5. Train the Model
```bash
python main.py
```
This will generate `voice_model.pkl` in your project folder.

### 6. Run the Web App
```bash
cd voice-access-web
python app.py
```
Open your browser → `http://127.0.0.1:5000`

---

## Usage

1. Open `http://127.0.0.1:5000` in your browser
2. Click Start Recording
3. Speak clearly for 3 seconds
4. Wait for the result:
   - **Access Granted** — your voice matched an authorized user
   - **Access Denied** — voice not recognized

---

## Authorized Users

By default the following users are authorized (edit in `app.py`):
```python
AUTHORIZED_USERS = ['user1', 'admin', 'authorized_user']
```
Change this list to match your trained user names.

---

## Model Details

| Parameter | Value |
|---|---|
| Algorithm | Support Vector Machine (SVM) |
| Kernel | Linear |
| Features | MFCC (13 coefficients) |
| Train/Test Split | 80% / 20% |
| Samples per User | 10 WAV files |
| Total Users | 5 |

---

## UI Features

- Live voice recording via browser
- Animated sound wave while recording
- Spinner while processing
- Dark / Light mode toggle
- Responsive design

---

## Dependencies

```
flask
librosa
scikit-learn
numpy
soundfile
```

Install all at once:
```bash
pip install flask librosa scikit-learn numpy soundfile
```
