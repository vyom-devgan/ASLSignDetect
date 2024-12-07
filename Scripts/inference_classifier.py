import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the pre-trained model for ASL character prediction
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize webcam for real-time video feed
cap = cv2.VideoCapture(0)

# Initialize Mediapipe's hands solution and related utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure the Mediapipe Hands module for detection
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Mapping of model output indices to ASL alphabet characters
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
               20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Variable to store the concatenated ASL characters forming a word
saved_word = ""

# Start video capture and processing loop
while True:
    data_aux = []  # Temporary storage for normalized hand landmarks
    x_ = []        # List to store x-coordinates of landmarks
    y_ = []        # List to store y-coordinates of landmarks

    # Capture a single frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions and convert to RGB for Mediapipe processing
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks in the frame
    results = hands.process(frame_rgb)
    predicted_character = ""

    if results.multi_hand_landmarks:
        # Draw detected hand landmarks on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Process hand landmarks for prediction
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize landmark coordinates relative to the bounding box
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Compute bounding box dimensions around the detected hand
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Predict the ASL character based on processed landmarks
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Draw the bounding box and display the predicted character
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Display the concatenated saved word on the frame
    cv2.putText(frame, f"Saved Word: {saved_word}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the video feed with annotations
    cv2.imshow('ASL Detector', frame)

    # Handle keypress events
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Append the predicted character to the saved word
        if predicted_character:
            saved_word += predicted_character
    elif key == ord('r'):  # Reset the saved word
        saved_word = ""
    elif key == ord('s'):  # Save the saved word to a file
        with open("saved_word.txt", "w") as file:
            file.write(saved_word)
    elif key == ord('q'):  # Quit the application
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()