# Sign Language Translator

[![Sign Language Translator Demo](https://img.youtube.com/vi/pspCDx4ifFk/0.jpg)](https://www.youtube.com/watch?v=pspCDx4ifFk)

This project implements a real-time sign language translator that uses **OpenCV**, **MediaPipe**, and a trained neural network to recognize American Sign Language (ASL) hand gestures from a live video feed.

## Features

- **Real-time hand gesture recognition**: Uses a webcam feed to recognize and classify ASL letters.
- **Custom-trained neural network**: A machine learning model trained to classify ASL letters (A-Z) as well as special gestures for `'del'`, `'nothing'`, and `'space'`.
- **Hand landmark detection**: Uses MediaPipe to detect 21 key points on the hand and extract coordinates for gesture classification.
- **Interactive visualization**: Displays the recognized hand gestures on the video feed in real-time.

## Dataset

- The dataset consists of images of hand gestures from the [**ASL Alphabet dataset**](https://www.kaggle.com/datasets/grassknoted/asl-alphabet). The dataset was split into training and test sets, with the landmarks of each image extracted using MediaPipe.
- Each image corresponds to a specific ASL letter or a gesture (`'del'`, `'nothing'`, `'space'`).

## Model

- The neural network was built using **TensorFlow/Keras** and consists of multiple dense layers:
  - **Input Layer**: Takes the 42 hand landmark features (21 points, each with x and y coordinates).
  - **Hidden Layers**: Two hidden layers with ReLU activations and dropout regularization to prevent overfitting.
  - **Output Layer**: Softmax output to classify the gesture into one of the 29 categories (A-Z, `'del'`, `'nothing'`, `'space'`).

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/dylen400mh/sign-language-translator.git
    ```

2. Navigate to the project directory:

    ```bash
    cd sign-language-translator
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Model Training

1. **Preprocessing**:
   - The dataset was preprocessed by extracting hand landmarks using MediaPipe.
   - The landmarks and labels were saved into CSV files (`train_landmarks.csv`, `test_landmarks.csv`).

2. **Model Architecture**:
   - Input: 42 hand landmark features (21 keypoints, each with x and y coordinates).
   - Hidden layers: Dense layers with 512, 256, and 128 units respectively, with ReLU activations and dropout regularization.
   - Output: Softmax layer with 29 output classes (A-Z, `'del'`, `'nothing'`, `'space'`).
  
3. **Training**:
  - The model was trained for 50 epochs using the Adam optimizer and sparse categorical cross-entropy loss.
  - Early stopping was applied to prevent overfitting.

## Usage

- Run the project with the following command:
  
   ```bash
    python main.py
    ```

- Once the project is running, it will open the webcam feed and display the recognized ASL hand gestures in real time.
- The recognized gesture will be shown on the top-left corner of the video feed.

## File Structure
```
sign-language-translator/
│
├── archive/                         # Contains the ASL Alphabet dataset
├── processed_data/                  # Preprocessed landmark data (CSV files)
├── models/                          # Trained model files
├── main.py                          # Main file for real-time gesture recognition
├── preprocess_dataset.py            # Preprocessing script for extracting hand landmarks
├── train_model.py                   # Script for training the neural network
└── README.md                        # Project documentation
```

## Acknowledgments
- The [**ASL Alphabet dataset**](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) is publicly available and was used for training the gesture recognition model.
- The hand tracking and landmark detection was powered by **MediaPipe** from Google.
