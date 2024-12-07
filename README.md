
# ASLSignDetect: American Sign Language Alphabet Detection

## Project Description
ASLSignDetect is an AI-powered image classification system designed to detect and recognize individual letters of the American Sign Language (ASL) alphabet. The project leverages machine learning techniques to process images of hand signs and predict the corresponding letter from the ASL alphabet. This system can be used to facilitate communication with individuals who are deaf or hard of hearing, enabling a more inclusive and accessible environment.

## Features
- **Real-time Detection**: Recognizes ASL hand signs in real-time using camera input.
- **High Accuracy**: Built with a Random Forest Classifier for high classification accuracy.
- **User-friendly Interface**: Simple interface that displays the predicted ASL letter.
- **Training and Testing**: Supports both model training on custom datasets and testing on unseen images.
- **Customizable**: Easily extensible to include more complex models or datasets.

## Functionality
The ASLSignDetect project includes the following core functionalities:
1. **Image Preprocessing**: Normalizes images for training and inference to ensure consistent input for the model.
2. **Model Training**: Uses a Random Forest Classifier to learn from a dataset of labeled ASL alphabet images.
3. **Prediction**: Makes predictions based on input images using the trained model.
4. **Model Evaluation**: Measures model performance with accuracy metrics and prints the classification score.

## Steps to Use
### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/yourusername/ASLSignDetect.git
cd ASLSignDetect
```

### 2. Install Dependencies
Install the required Python libraries using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset
You can use a pre-existing dataset of ASL hand signs or prepare your own. The dataset should contain images of ASL alphabet letters, where each image is labeled according to the letter it represents.

- For testing, you can download the ASL Alphabet dataset from [link to dataset].

### 4. Train the Model
To train the model, run the following Python script:
```bash
python train_model.py
```
This will process the dataset, train the Random Forest model, and save the trained model to a file for later use.

### 5. Run the ASL Detection
To run the detection system, use:
```bash
python detect_asl.py
```
This will start the camera, and the system will display the predicted letter for any ASL hand sign it detects.

### 6. Evaluate the Model
Once the model is trained, you can evaluate its performance using the following command:
```bash
python evaluate_model.py
```
This will display the accuracy of the model on a test set.

## Requirements

- Python 3.7+
- Libraries:
  - `numpy`
  - `scikit-learn`
  - `opencv-python`
  - `matplotlib`
  - `pickle`

### `requirements.txt`:
```txt
numpy==1.21.0
scikit-learn==0.24.2
opencv-python==4.5.1.48
matplotlib==3.4.2
pickle-mixin==1.0.2
```

## Directory Structure
The project follows this directory structure:
```
ASLSignDetect/
├── data/                 # Dataset folder (e.g., ASL alphabet images)
├── model/                # Trained model file
├── scripts/              # Python scripts for training and testing
│   ├── train_model.py    # Script to train the model
│   ├── detect_asl.py     # Script for real-time detection
│   └── evaluate_model.py # Script to evaluate model accuracy
├── requirements.txt      # Required dependencies
└── README.md             # This file
```

## Future Enhancements
- **Real-time Webcam Integration**: Integrate with live webcam feeds for real-time sign language recognition.
- **Deep Learning Models**: Explore deep learning models such as Convolutional Neural Networks (CNNs) for improved accuracy.
- **Multi-sign Recognition**: Extend the model to recognize sequences of signs for word or sentence-level recognition.
- **Mobile App**: Create a mobile app for both iOS and Android that can use the trained model for ASL recognition on mobile devices.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

If you have any questions or suggestions, feel free to open an issue or contribute to the project.
