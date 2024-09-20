import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# initialize mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# PREPROCESS TRAINING DATASET
train_dataset_path = "archive\\asl_alphabet_train\\asl_alphabet_train"
train_labels = []
train_landmark_list = []

# loop through dataset folders (for each letter)
for gesture_folder in os.listdir(train_dataset_path):
    gesture_folder_path = os.path.join(train_dataset_path, gesture_folder)

    print(gesture_folder_path)
    for image_file in os.listdir(gesture_folder_path):
        image_path = os.path.join(gesture_folder_path, image_file)
        print(image_path)
        img = cv2.imread(image_path)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # extract landmarks (x,y) then flatten
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y])
                landmarks = np.array(landmarks).flatten()
                train_landmark_list.append(landmarks)
                train_labels.append(gesture_folder)

# perform 80-20 split using train_test_split
all_landmarks = np.array(train_landmark_list)
all_labels = np.array(train_labels)

X_train, X_test, y_train, y_test = train_test_split(all_landmarks, all_labels, test_size=0.2, random_state=42)

# save training landmarks and labels to csv
train_df = pd.DataFrame(X_train)
train_df['label'] = y_train
train_df.to_csv('processed_data/train_landmarks.csv', index=False)

# Save test landmarks and labels to a CSV file
test_df = pd.DataFrame(X_test)
test_df['label'] = y_test
test_df.to_csv("processed_data/test_landmarks.csv", index=False)
