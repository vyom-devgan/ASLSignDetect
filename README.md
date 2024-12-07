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
pip install -r requirements.txt
