# Facial Emotion Recognition System

A real-time facial emotion recognition system using OpenCV and a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

## Project Structure

```text
emotion_recognition/
├── data/               # Place FER2013 dataset here (train/ and test/ folders)
├── models/             # Contains the trained model (emotion_model.h5)
├── haarcascades/       # (Optional) Place for Haar Cascade XML files
├── train.py            # Script to train the CNN model
├── detect.py           # Script for real-time webcam detection
├── requirements.txt    # Python dependencies
└── README.md           # Instructions for use
```

## Features
- Real-time webcam capture using OpenCV.
- Face detection using Haar Cascades.
- Emotion classification into 7 categories: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
- CNN model architecture optimized for FER2013 dataset.

## Installation

1. **Clone or download** this repository.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preparation
- Download the FER2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).
- Extract the dataset into the `data/` folder. It should have `train/` and `test/` subdirectories, each containing folders for the 7 emotion classes.

### 2. Training the Model
- Run the training script:
  ```bash
   python train.py
   ```
- This will train the CNN and save the trained weights as `models/emotion_model.h5`.

### 3. Real-time Detection
- Once the model is trained, start the detector:
  ```bash
   python detect.py
   ```
- A window will open showing your webcam feed with emotion labels.
- Press **'q'** to exit the application.

## Model Details
- **Input Size**: 48x48 pixels (Grayscale).
- **Architecture**: 3 Convolutional blocks with Batch Normalization, MaxPooling, and Dropout, followed by a Dense layer and a Softmax output.
- **Accuracy**: Dependent on training epochs and data quality.
