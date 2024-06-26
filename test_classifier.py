import pickle
import cv2
import mediapipe as mp
import numpy as np

# Grabs our model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)  # Start the camera object

# Mediapipe objects to identify hand landmarks and draw out landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Detect hands and create a model using mp_hands 
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# To convert our labels to sign language
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read() # Access camera

    H, W, _ = frame.shape

    # Mediapipe requires images to be in rgb so we must convert
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect all the landmarks of the rgb image using hands model
    results = hands.process(frame_rgb)

    # If you detect a hand
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw out the hand landmarks 
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())


        for hand_landmarks in results.multi_hand_landmarks: # Iterate through hands
            for i in range(len(hand_landmarks.landmark)): # Iterate through landmarks in hand
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                data_aux.append(x)
                data_aux.append(y)
                
                x_.append(x)
                y_.append(y)




        # Calculations for a box around the hand
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10 # Find bottom corner of hand rectangle

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # predict our class using data aux x and y
        prediction = model.predict([np.asarray(data_aux)]) 

        # Convert prediction into sign language class
        predicted_character = labels_dict[int(prediction[0])]


        # Draw out out prediction
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4) # 000 is black colour, 4 is thickness
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA) # Paste our prediction

    cv2.imshow('frame', frame) # Show our final processed frame
    cv2.waitKey(1) # Wait a second until next frame


cap.release()  # Release the camera from memory
cv2.destroyAllWindows()
