import os
# To save data/datasets
import pickle
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

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        # Mediapipe requires images to be in rgb so we must convert
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect all the landmarks of the rgb image using hands model
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks: # If you detect a hand
            for hand_landmarks in results.multi_hand_landmarks: # Loop through each detected hand
                
                # Create an array of xyz values of landmarks to train the classifier with
                for i in range(len(hand_landmarks.landmark)): # Loop through each landmark in hand
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            data.append(data_aux) # Append a new array for each class
            labels.append(dir_)   # Append class number 

        # Matplotlib also requires images in rgb 
#         plt.figure()
#         # Plot the images in x and y axis
#         plt.imshow(img_rgb)

# plt.show()

file = open('data.pickle', 'wb') # open new file in binary
pickle.dump({'data': data, 'labels': labels}, file) # Create a dictionary with data arrays and labels
file.close()
