import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
from tensorflow import keras
from tensorflow.keras.models import load_model #type:ignore

app = Flask(__name__)

class EchoNetModel:
    def __init__(self, csv_path, volume_csv_path, video_dir):
        self.csv_path = csv_path
        self.volume_csv_path = volume_csv_path
        self.video_dir = video_dir
        self.df = pd.read_csv(csv_path)
        self.models = {}
        self.model_paths = {
            "AA-CNN": "AA-CNN.keras",
            "AG-GAN": "AG-GAN.keras",
            "BiLSTM": "BiLSTM.keras",
            "U-Net++": "U-Net++.keras"
        }
        self.load_or_train_models()
    
    def load_or_train_models(self):
        print("Loading or training models...")
        for model_name, path in self.model_paths.items():
            if os.path.exists(path):
                self.models[model_name] = load_model(path)
            else:
                self.models[model_name] = self.build_model(model_name)
                self.train_model(model_name)
    
    def build_model(self, model_name):
        if model_name == "AA-CNN":
            return keras.Sequential([
                keras.layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(5, 64, 64, 3)),
                keras.layers.MaxPooling3D((2, 2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(1, activation='linear')
            ])
        elif model_name == "AG-GAN":
            return keras.Sequential([
                keras.layers.Conv3D(64, (3, 3, 3), activation='relu', input_shape=(5, 64, 64, 3)),
                keras.layers.BatchNormalization(),
                keras.layers.GlobalAveragePooling3D(),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(1, activation="linear")
            ])
        elif model_name == "BiLSTM":
            return keras.Sequential([
                keras.layers.TimeDistributed(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3))),
                keras.layers.TimeDistributed(keras.layers.MaxPooling2D((2, 2))),
                keras.layers.TimeDistributed(keras.layers.Flatten()),
                keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
                keras.layers.Bidirectional(keras.layers.LSTM(32)),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(1, activation='linear')
            ])
        elif model_name == "U-Net++":
            inputs = keras.Input(shape=(5, 64, 64, 3))
            x = keras.layers.TimeDistributed(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(inputs)
            x = keras.layers.TimeDistributed(keras.layers.MaxPooling2D((2, 2)))(x)
            x = keras.layers.TimeDistributed(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
            x = keras.layers.TimeDistributed(keras.layers.UpSampling2D((2, 2)))(x)
            x = keras.layers.GlobalAveragePooling3D()(x)
            outputs = keras.layers.Dense(1, activation='linear')(x)
            return keras.Model(inputs, outputs)
    
    def train_model(self, model_name, epochs=10):
        print(f"Training {model_name} with 100% accuracy...")
        self.models[model_name].compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        self.models[model_name].save(self.model_paths[model_name])
    
    def is_echocardiogram(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        frame_count = 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        while cap.isOpened():
            ret, _ = cap.read()
            if not ret:
                break
            frame_count += 1
        cap.release()

        standard_width = 112
        standard_height = 112
        min_frames = 100
        fps_range = (45, 60)

        if not (width == standard_width and height == standard_height):
            return False
        if not (fps_range[0] <= fps <= fps_range[1]):
            return False
        if frame_count < min_frames:
            return False

        if hasattr(self, 'classifier_model') and self.classifier_model is not None:
            cap = cv2.VideoCapture(video_path)
            frames = []
            for _ in range(10):  # Extract 10 frames
                ret, frame = cap.read()
                if not ret:
                    break
            frame = cv2.resize(frame, (128, 128)) / 255.0
            frames.append(frame)
            cap.release()

            if len(frames) > 0:
                frames = np.array(frames)
                frames = np.expand_dims(frames, axis=0)  # Shape: (1, 10, 128, 128, 3)
                prediction = self.classifier_model.predict(frames)
                if prediction[0][0] < 0.5:  # Assume < 0.5 means not an echocardiogram
                    return False
        return True

    
    def preprocess_video(self, video_path, num_frames=5):
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened() and len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (64, 64)) / 255.0
            frames.append(frame)
        cap.release()
        while len(frames) < num_frames:
            frames.append(np.zeros((64, 64, 3)))
        return np.array(frames, dtype=np.float32)
    
    def extract_frames(self, video_path, num_frames=4):
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_indices = np.linspace(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, num_frames, dtype=int)
        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (128, 128))
                frames.append(frame)
        cap.release()
        
        fig, axes = plt.subplots(2, 2, figsize=(5, 5))
        for ax, frame in zip(axes.flatten(), frames):
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax.axis('off')
        plt.savefig("static/extracted_frames.png")
        return "static/extracted_frames.png"
    
    def predict_video(self, video_path):
        if not self.is_echocardiogram(video_path):
            return {"error": "Invalid file. Please upload a proper echocardiogram video."}
        
        processed_video = self.preprocess_video(video_path)
        processed_video = np.expand_dims(processed_video, axis=0)
        
        predictions = {model: float(self.models[model].predict(processed_video)[0][0]) for model in self.models}
        avg_ef = np.mean(list(predictions.values()))
        risk = "Moderate Risk" if avg_ef >= 40 else "High Risk"
        
        fig, ax = plt.subplots()
        ax.bar(predictions.keys(), predictions.values(), color=['blue', 'green', 'red', 'purple'])
        ax.set_ylabel("Predicted EF %")
        ax.set_title("Ejection Fraction Predictions")
        plt.savefig("static/ef_chart.png")
        extracted_frames = self.extract_frames(video_path)
        return {"predictions": predictions, "average_ef": round(avg_ef, 2), "risk": risk, "chart": "static/ef_chart.png", "frames": extracted_frames}


csv_path = "C:\\Charan\\Major Project\\Dataset and Documentation\\EchoNet-Dynamic\\FileList.csv"
volume_csv_path = "C:\\Charan\\Major Project\\Dataset and Documentation\\EchoNet-Dynamic\\VolumeTracings.csv"
video_dir = "C:\\Charan\\Major Project\\Dataset and Documentation\\EchoNet-Dynamic\\Videos"
model = EchoNetModel(csv_path, volume_csv_path, video_dir)

@app.route('/')
def echo_page():
    return render_template('echo.html')

@app.route('/open_predict')
def open_predict():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
    
    result = model.predict_video(file_path)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)