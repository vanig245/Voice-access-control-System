from flask import Flask, request, render_template
import librosa
import numpy as np
import pickle
import os
import subprocess

app = Flask(__name__)
model = pickle.load(open("voice_model.pkl", "rb"))

AUTHORIZED_USERS = ['user1', 'admin', 'authorized_user']


def extract_features(file_path):
    try:
        print(f"Reading audio from: {file_path}")
        y, sr = librosa.load(file_path, sr=None)
        # y, sr = sf.read(file_path)
        print(f"Audio loaded: shape={y.shape}, sample_rate={sr}")

       
        if len(y.shape) > 1:
            y = y.mean(axis=1)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        print(f"MFCC extracted: shape={mfcc.shape}")
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print("Feature extraction error:", e)
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        audio = request.files["audio"]
        webm_path = "temp.webm"
        wav_path = "temp.wav"

        audio.save(webm_path)

        subprocess.run(["ffmpeg", "-y", "-i", webm_path, wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        features = extract_features(wav_path)
        if features is None:
            return "Error processing audio features"

        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]

        if prediction in AUTHORIZED_USERS:
            return "✅ Access Granted"
        else:
            return "❌ Access Denied"
    except Exception as e:
        print("Prediction error:", e)
        return f"Error: {str(e)}"



if __name__ == "__main__":
    app.run(debug=True)
