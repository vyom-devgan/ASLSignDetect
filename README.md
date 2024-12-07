
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

### 4. Steps to use

1. **Run `collect_imgs.py`**  
   Collect images for each ASL letter by following the instructions within the script. The images will be saved in respective folders for each letter.

2. **Run `create_dataset.py`**  
   Create a dataset by labeling and organizing the collected images for training.

3. **Run `train_classifier.py`**  
   Train the classifier using the labeled dataset. This will output a trained model that can recognize ASL signs.

4. **Run `inference_classifier.py`**  
   Use the trained classifier to detect and recognize ASL signs in real-time.

### Special Keys in `inference_classifier.py`

While using `inference_classifier.py`, there are special keys you can press for different actions:

- **'Q'**: Quit the application.
- **'SPACEBAR'**: Save the current letter being detected.
- **'S'**: Save the current word. This is saved to `saved_word.txt` but will overwrite the previous word.
- **'R'**: Reset the current word (clear the current word being built).

## Example of Running the Scripts

1. **Collect Images for Each Letter**:
    Run the `collect_imgs.py` script and follow the prompts to collect images for each ASL letter.

    ```bash
    python Scripts/collect_imgs.py
    ```

2. **Create Dataset**:
    Once the images are collected, run the `create_dataset.py` script to create the dataset.

    ```bash
    python Scripts/create_dataset.py
    ```

3. **Train the Classifier**:
    After the dataset is prepared, train the classifier with the following command:

    ```bash
    python Scripts/train_classifier.py
    ```

4. **Inference (Real-Time Prediction)**:
    Once the model is trained, run the `inference_classifier.py` to start recognizing ASL signs in real-time.

    ```bash
    python Scripts/inference_classifier.py
    ```

## requirements.txt

This file lists the necessary libraries without specific versions. Install them using:

```bash
pip install -r requirements.txt


## Requirements

- Python 3.7+
- Libraries:
  - `numpy`
  - `scikit-learn`
  - `opencv-python`
  - `matplotlib`
  - `pickle`

### `requirements.txt`:
```bash
opencv-python
mediapipe
scikit-learn
```

## Directory Structure
The project follows this directory structure:
```bash
ASLSignDetect/
├── Scripts/
│   ├── collect_imgs.py        # Collect images of ASL signs
│   ├── create_dataset.py      # Create a dataset from collected images
│   ├── train_classifier.py    # Train the classifier
│   ├── inference_classifier.py# Use the trained model for inference
├── requirements.txt           # Required dependencies
└── README.md                  # Project overview
```
