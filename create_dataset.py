import os
# To create landmarks
import mediapipe as mp
# To collect data from the camera
import cv2
# To plot images
import matplotlib.pyplot as plt

# Mediapipe objects to identify hand landmarks and draw out landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Detect hands and create a model using mp_hands 
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_))[:1]:
        data_aux = []

        x_ = []
        y_ = []

        # Mediapipe requires images to be in rgb so we must convert
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect all the landmarks of the rgb image using hands model
        results = hands.process(img_rgb)

        # Draw out the landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img_rgb, # img to draw
                hand_landmarks, # The landmark identified in results
                mp_hands.HAND_CONNECTIONS, # Hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Matplotlib also requires images in rgb 
        plt.figure()
        # Plot the images in x and y axis
        plt.imshow(img_rgb)

plt.show()