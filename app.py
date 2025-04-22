import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import imageio
import cv2 # type: ignore
import matplotlib.pyplot as plt

# Load dataset
csv_path = "C:\\Users\\Admin\\Downloads\\EchoNet-Dynamic\\FileList.csv"
df = pd.read_csv(csv_path)
print(f"Total videos for training: {len(df)}")

def preprocess_video(video_path, num_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (112, 112))  # Resize to match dataset dimensions
        frame = frame / 255.0  # Normalize pixel values
        frames.append(frame)
    cap.release()
    while len(frames) < num_frames:
        frames.append(np.zeros((112, 112, 3)))  # Pad with black frames if needed
    return np.array(frames)

# Load and preprocess dataset
video_paths = glob("C:\\Users\\Admin\\Downloads\\EchoNet-Dynamic\\Videos\\*.avi")
X, y = [], []

for video_path in tqdm(video_paths[:100]):  # Limit to 100 videos for testing
    video_id = os.path.basename(video_path).split('.')[0]
    if video_id in df['FileName'].values:
        label = df[df['FileName'] == video_id]['EF'].values[0]  # Example: Ejection Fraction
        X.append(preprocess_video(video_path))
        y.append(label)

X = np.array(X)
y = np.array(y)

# Attention-Augmented CNN for View Classification
def build_aacnn():
    model = keras.Sequential([
        keras.Input(shape=(10, 112, 112, 3)),  # Fixed number of frames
        layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='linear')  # Regression for EF prediction
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train and Evaluate Models
view_classification_model = build_aacnn()
view_classification_model.fit(X, y, epochs=10, batch_size=8, validation_split=0.2)

# Function to process new user input video and predict disease
def predict_new_video(video_path):
    processed_video = preprocess_video(video_path)
    processed_video = np.expand_dims(processed_video, axis=0)  # Add batch dimension
    prediction = view_classification_model.predict(processed_video)
    return prediction[0][0]

# User input for video path
user_video_path = input("Enter the path to the echocardiogram video: ")
result = predict_new_video(user_video_path)
print(f"Predicted Ejection Fraction: {result:.2f}")

# Interpretation of the result
def interpret_prediction(prediction):
    if prediction < 40:
        return "High risk of heart disease - Seek medical attention."
    elif 40 <= prediction < 55:
        return "Moderate risk - Further evaluation recommended."
    else:
        return "Normal range - No immediate concerns."

interpretation = interpret_prediction(result)
print(f"Interpretation: {interpretation}")
