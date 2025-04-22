import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate # type: ignore

class EchoNetModel:
    def __init__(self, csv_path, video_dir):
        self.csv_path = csv_path
        self.video_dir = video_dir
        self.df = pd.read_csv(csv_path)
        self.aacnn_model = None
        self.aggan_model = None
        self.bilstm_model = None
        self.unetpp_model = None

    def preprocess_video(self, video_path, num_frames=10):
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened() and len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112)) / 255.0
            frames.append(frame)
        cap.release()

        while len(frames) < num_frames:
            frames.append(np.zeros((112, 112, 3)))
        return np.array(frames)

    def extract_and_display_frames(self, video_path, num_frames=4):
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened() and len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)
        cap.release()

        if len(frames) < num_frames:
            print("❌ Not enough frames extracted.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
            ax.axis("off")

        plt.suptitle("Extracted Video Frames", fontsize=14)
        plt.show()

    def build_aacnn(self):
        model = keras.Sequential([
            layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(10, 112, 112, 3)),
            layers.MaxPooling3D((2, 2, 2)),
            layers.Conv3D(64, (3, 3, 3), activation='relu'),
            layers.MaxPooling3D((2, 2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def build_bilstm(self):
        model = keras.Sequential([
            layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(10, 112, 112, 3))),
            layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
            layers.TimeDistributed(layers.Flatten()),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def build_aggan(self):
        model = keras.Sequential([
            layers.Conv3D(64, (3, 3, 3), activation='relu', input_shape=(10, 112, 112, 3)),
            layers.BatchNormalization(),
            layers.Conv3D(128, (3, 3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling3D(),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="linear")
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def build_unetpp(self, input_shape=(112, 112, 3)):
        inputs = Input(input_shape)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        pool1 = MaxPooling2D((2, 2))(conv1)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        pool2 = MaxPooling2D((2, 2))(conv2)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
        up1 = UpSampling2D((2, 2))(conv3)
        merge1 = Concatenate()([conv2, up1])
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge1)
        up2 = UpSampling2D((2, 2))(conv4)
        merge2 = Concatenate()([conv1, up2])
        conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge2)
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv5)
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_models(self):
        X, y = np.random.rand(500, 10, 112, 112, 3), np.random.rand(500)
        X_seg, y_seg = np.random.rand(500, 112, 112, 3), np.random.randint(0, 2, (500, 112, 112, 1))

        print("\n🚀 Training AA-CNN (View Classification)...")
        self.aacnn_model = self.build_aacnn()
        self.aacnn_model.fit(X, y, epochs=5, batch_size=16, validation_split=0.2)

        print("\n🚀 Training AG-GAN (Disease Detection)...")
        self.aggan_model = self.build_aggan()
        self.aggan_model.fit(X, y, epochs=5, batch_size=16, validation_split=0.2)

        print("\n🚀 Training BiLSTM (Temporal Prognosis)...")
        self.bilstm_model = self.build_bilstm()
        self.bilstm_model.fit(X, y, epochs=5, batch_size=16, validation_split=0.2)

        print("\n🚀 Training U-Net++ (Segmentation)...")
        self.unetpp_model = self.build_unetpp()
        self.unetpp_model.fit(X_seg, y_seg, epochs=5, batch_size=16, validation_split=0.2)

    def predict_video(self, video_path):
        processed_video = self.preprocess_video(video_path)
        processed_video = np.expand_dims(processed_video, axis=0)

        predictions = {
            "AA-CNN (View Classification)": self.aacnn_model.predict(processed_video)[0][0],
            "AG-GAN (Disease Detection)": self.aggan_model.predict(processed_video)[0][0],
            "BiLSTM (Temporal Prognosis)": self.bilstm_model.predict(processed_video)[0][0]
        }

        return {model: self.interpret_risk(value) for model, value in predictions.items()}

    def interpret_risk(self, ef_value):
        if ef_value >= 55:
            return "Normal"
        elif 45 <= ef_value < 55:
            return "Borderline"
        elif 30 <= ef_value < 45:
            return "Moderate Risk"
        else:
            return "High Risk"

def plot_results(predictions):
    colors = {"Normal": "green", "Borderline": "blue", "Moderate Risk": "orange", "High Risk": "red"}

    plt.figure(figsize=(6, 4))
    risk_labels = list(predictions.values())
    models = list(predictions.keys())
    color_map = [colors[label] for label in risk_labels]

    plt.bar(models, [1] * len(models), color=color_map)
    plt.xlabel("Models")
    plt.ylabel("Risk Level")
    plt.title("Ejection Fraction Risk Interpretation")

    for i, label in enumerate(risk_labels):
        plt.text(i, 1.02, label, ha='center', fontsize=12, weight='bold')

    plt.ylim(0, 1.5)
    plt.xticks(rotation=15)
    plt.show()

csv_path = "C:\\Users\\Admin\\Downloads\\EchoNet-Dynamic\\FileList.csv"
video_dir = "C:\\Users\\Admin\\Downloads\\EchoNet-Dynamic\\Videos\\"
echonet = EchoNetModel(csv_path, video_dir)
echonet.train_models()

while True:
    video_path = input("\n🎥 Enter video path (or type 'exit' to quit): ")
    if video_path.lower() == "exit":
        break
    predictions = echonet.predict_video(video_path)
    print("\n📊 Predictions:", predictions)
