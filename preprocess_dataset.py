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
train_landmarks_df.to_csv('processed_data/train_landmarks.csv', index=False)

# PREPROCESS TEST DATASET
test_dataset_path = 'archive\\asl_alphabet_test\\asl_alphabet_test'
test_labels = []
test_landmark_list = []

# Loop through the test set folders
for image_file in os.listdir(test_dataset_path):
    image_path = os.path.join(test_dataset_path, image_file)
    img = cv2.imread(image_path)

    # Check if the file is 'del', 'nothing', or 'space', otherwise take the first character
    if 'del' in image_file:
        label = 'del'
    elif 'nothing' in image_file:
        label = 'nothing'
    elif 'space' in image_file:
        label = 'space'
    else:
        label = image_file[0]  # e.g., from 'A1234' it extracts 'A'

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
            test_labels.append(label)

# Save test landmarks and labels to a CSV file
test_landmarks_df = pd.DataFrame(test_landmark_list)
test_landmarks_df['label'] = test_labels
test_landmarks_df.to_csv("processed_data/test_landmarks.csv", index=False)