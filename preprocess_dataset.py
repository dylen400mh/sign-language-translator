import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd

# initialize mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

#PREPROCESS TRAINING DATASET
train_dataset_path = 'archive/asl_alphabet_train'
train_labels = []
train_landmark_list = []

# loop through dataset folders (for each letter)
for gesture_folder in os.listdir(train_dataset_path):
    gesture_folder_path = os.path.join(train_dataset_path, gesture_folder)

    for image_file in os.listdir(gesture_folder_path):
        image_path = os.path.join(gesture_folder_path, image_file)
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                #extract landmarks (x,y) then flatten
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y])
                landmarks = np.array(landmarks).flatten()
                train_landmark_list.append(landmarks)
                train_labels.append(gesture_folder)

# save training landmarks and labels to csv
train_landmarks_df = pd.DataFrame(train_landmark_list)
train_landmarks_df['Label'] = train_labels
train_landmarks_df.to_csv('train_landmarks.csv', index=False)

# PREPROCESS TEST DATASET
test_dataset_path = "archive/asl_alphabet_test"
test_labels = []
test_landmark_list = []

# Loop through the test set folders
for image_file in os.listdir(test_dataset_path):
    image_path = os.path.join(test_dataset_path, image_file)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            #extract landmarks (x,y) then flatten
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y])
            landmarks = np.array(landmarks).flatten()
            test_landmark_list.append(landmarks)

            #extract letter from filenames
            test_labels.append(image_file.split('.')[0])

# Save test landmarks and labels to a CSV file
test_landmarks_df = pd.DataFrame(test_landmark_list)
test_landmarks_df['label'] = test_labels
test_landmarks_df.to_csv("test_landmarks.csv", index=False)