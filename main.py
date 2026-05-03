import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

def extract_features(file):
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

X, y = [], []
dataset_path = "voice_data"

for user in os.listdir(dataset_path):
    user_folder = os.path.join(dataset_path, user)
    if os.path.isdir(user_folder):
        for filename in os.listdir(user_folder):
            if filename.endswith(".wav"):
                file_path = os.path.join(user_folder, filename)
                features = extract_features(file_path)
                X.append(features)
                y.append(user)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Save model
with open("voice_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as voice_model.pkl")
