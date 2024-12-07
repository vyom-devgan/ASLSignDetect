import os
import pickle
import mediapipe as mp
import cv2

# Initialize Mediapipe modules for hand detection and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the Hands module with specified parameters
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory containing the dataset
DATA_DIR = './data'

# Lists to store processed data and corresponding labels
data = []
labels = []

# Iterate over each class folder in the dataset directory
for dir_ in os.listdir(DATA_DIR):
    # Iterate over each image file within the class folder
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Temporary list to store hand landmark data
        x_ = []        # Temporary list for x-coordinates of landmarks
        y_ = []        # Temporary list for y-coordinates of landmarks

        # Read the image from the dataset and convert it to RGB format
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image using Mediapipe to detect hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # Iterate over detected hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract x and y coordinates of each landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize the x and y coordinates and append to data_aux
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Append processed landmark data and label to respective lists
            data.append(data_aux)
            labels.append(dir_)

# Save the processed data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)