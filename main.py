import cv2
import mediapipe as mp

# setup camera
cam = cv2.VideoCapture(0)

# initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

while True:
    # capture frame-by-frame
    ret, frame = cam.read()
    # flip frame horizontally (so mouse goes right direction)
    frame = cv2.flip(frame, 1)
    # get frame dimensions
    h, w, _ = frame.shape

    # detect hand in frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # process frame and detect hand
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # draw landmarks on hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # display frame
    cv2.imshow('Sign Language Translator', frame)

    # break loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release camera and close windows
cam.release()
cv2.destroyAllWindows()
