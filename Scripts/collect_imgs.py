import os
import cv2

# Define the directory for storing the dataset
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Set the number of classes and the dataset size per class
number_of_classes = 26
dataset_size = 100

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Iterate through each class to collect data
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q"! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        file_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(file_path, frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()