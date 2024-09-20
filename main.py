import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# load model
model = tf.keras.models.load_model('models/sign_language_model.h5')


# setup camera
cam = cv2.VideoCapture(0)

# initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Dictionary for class labels
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

while True:
    # capture frame-by-frame
    ret, frame = cam.read()
    # flip frame horizontally (so mouse goes right direction)
    frame = cv2.flip(frame, 1)

    # detect hand in frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # draw landmarks on hand
            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

            # extract hand landmarks as input for model
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y])

            # 21 landmarks expected
            if len(landmarks) == 21:
                # convert landmarks to numpy array and flatten
                landmarks = np.array(landmarks).flatten().reshape(1, -1)

                # predict gesture
                prediction = model.predict(landmarks)
                predicted_class = np.argmax(prediction)
                predicted_label = class_labels[predicted_class]

                # display prediction on screen
                cv2.putText(frame, f'Predicted Sign: {predicted_label}', (
                    50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                print("Error: Incorrect number of landmarks identified")
    # display frame
    cv2.imshow('Sign Language Translator', frame)

    # break loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release camera and close windows
cam.release()
cv2.destroyAllWindows()
