# Cats vs Dogs Classification

This repository contains code for classifying images of cats and dogs using a Convolutional Neural Network (CNN).

## Overview

The project consists of the following files:

*   `Cats_vs_Dogs_Classification.ipynb`: A Jupyter Notebook containing the code for building, training, and evaluating the CNN model.
*   `Streamlit_Code.py`: A Python script for creating a Streamlit web application to predict image classes.
*   `model_source_code.py`: Python script to define the CNN model.
*   `cat.jpg`: An example cat image for model testing.
*   `dog.jpg`: An example dog image for model testing.
*   `requirements.txt`: A list of Python packages required to run the code.

## Setup

1.  **Clone the repository:**

    ```
    git clone https://github.com/kinjal-mitra/CatsVSDogsClassification.git
    cd CatsVSDogsClassification
    ```

2.  **Create a virtual environment (recommended):**

    ```
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required packages:**

    ```
    pip install -r requirements.txt
    ```

4.  **Download the dataset:**

    *   Download the "dogs-vs-cats" dataset from [Kaggle](https://www.kaggle.com/datasets/salader/dogs-vs-cats). You may need to create a Kaggle account if you don't already have one.
    *   Place the downloaded ZIP file directly in the same directory as the code files (e.g., `Cats_vs_Dogs_Classification.ipynb`, `Streamlit_Code.py`).
    *   The code will automatically handle extracting the files from the ZIP.

## Usage

### 1. Training the Model

*   **Option 1: Using the Jupyter Notebook:**
    *   Open and run the `Cats_vs_Dogs_Classification.ipynb` notebook in Jupyter.
    *   Follow the instructions in the notebook to train the CNN model.
    *   **Running the notebook will save the trained TensorFlow model to a file** (e.g., `cat_dog_model.h5`). The notebook should specify where the model is saved.

*   **Option 2: Using the `model_source_code.py` file:**
    *   Run the `model_source_code.py` script. You might need to adjust the script to load the training data correctly, depending on how it handles reading from the ZIP file.
    *   **Running the script will save the trained TensorFlow model to a file** (e.g., `cat_dog_model.h5`). The script should specify where the model is saved.

### 2. Running the Streamlit App

1.  Make sure you have a trained model file (e.g., `cat_dog_model.h5`). This will be created by running either the Jupyter Notebook or the `model_source_code.py` file. Modify the `Streamlit_Code.py` script to point to the correct path of the model file.
2.  Run the Streamlit app:

    ```
    streamlit run Streamlit_Code.py
    ```

3.  Open the URL displayed in the terminal in your web browser.
4.  Upload an image of a cat or dog to the app.
5.  The app will display the predicted class (cat or dog).

## Model Architecture (If applicable, expand based on notebook content)

The CNN model consists of the following layers:

*   Convolutional layers
*   Max pooling layers
*   Fully connected layers
*   (Add more details as needed)

## Contributing

Feel free to contribute to this project by submitting pull requests.

