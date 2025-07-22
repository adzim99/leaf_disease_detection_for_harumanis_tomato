# A CNN-Based System for Plant Disease Detection via Image Classification

[](https://harumanis-tomato-disease-detection.streamlit.app/)

## 📖 Project Overview

This project provides a deep-learning solution to address the critical issue of plant disease identification in the agricultural sector. Agriculture is a vital component of economic growth and food security worldwide, yet crop yields are constantly threatened by diseases, leading to significant economic losses. This system aims to mitigate these threats by providing an accessible and accurate tool for early disease detection, aligning with the **United Nations' Sustainable Development Goal 2: Zero Hunger**.

-----

## 🎯 The Problem

Traditional methods for identifying plant diseases suffer from several key limitations that this project aims to solve:

  * **Inefficiency and Subjectivity**: Visual inspections by humans are often slow, subjective, and not suitable for large-scale farming operations.
  * **Inaccurate Treatments**: Inexperienced farmers may misidentify diseases, leading to the use of incorrect treatments, which increases costs and can cause environmental harm.
  * **Expert Shortage**: A lack of available agricultural experts and plant pathologists hinders effective and timely disease monitoring.
  * **Subtlety of Symptoms**: Early signs of disease can be subtle and difficult to detect, while symptoms can vary widely, requiring specialized tools for accurate diagnosis.
  * **Challenging Field Conditions**: Factors like inconsistent lighting and complex backgrounds in a natural field environment make manual detection difficult.

-----

## 📋 Objectives

The primary objectives of this project were:

1.  To review existing methods for plant disease detection and identify best practices for applying deep learning techniques.
2.  To develop a Convolutional Neural Network (CNN) based model capable of accurately detecting plant diseases from leaf images.
3.  To deploy the trained model on a web application for real-time use and evaluate its performance using key metrics like accuracy, precision, recall, and F1-score.

-----

## 🔬 Methodology & Model Architecture

The project followed a systematic workflow: **Data Acquisition** → **Model Development** → **Classification** → **Performance Evaluation** → **Deployment**. The core of the system is a custom-built Convolutional Neural Network (CNN).

### CNN Architecture

The model is composed of the following layers designed to learn and classify features from leaf images:

  * **Input Layer**: Accepts an image and normalizes its pixel values.
  * **3x Convolutional & Max-Pooling Blocks**: These layers work in sequence to learn hierarchical features, from low-level edges to complex patterns indicative of diseases. Max-pooling reduces the spatial dimensions, retaining the most important information.
  * **Flatten Layer**: Converts the final 2D feature map into a 1D vector.
  * **Dense Layer**: A fully connected layer that aggregates the features for prediction.
  * **Dropout Layer**: A regularization technique that randomly sets a fraction of input units to 0 at each update during training to prevent overfitting.
  * **Output Layer**: Uses a `softmax` function to classify the input image into one of the predefined disease classes.

-----

## 💾 Datasets

Two publicly available datasets from Kaggle were used for training and evaluation.

### 1\. Harumanis Mango Leaves

  * **Source**: Harumanis Mango Leaves Dataset 2021 by Gining et al.
  * **Classes**: Anthracnose, Black Sooty Mold, Healthy
  * **Data Distribution**:

| Set | Anthracnose | Black Sooty Mold | Healthy | Total |
| :--- | :--- | :--- | :--- | :--- |
| **Training** | 368 | 459 | 157 | 984 |
| **Validation** | 79 | 99 | 34 | 212 |
| **Testing** | 78 | 98 | 33 | 209 |
| **Total** | **525** | **656** | **224** | **1405** |

### 2\. Tomato Leaves

  * **Source**: Tomato Leaves, 2020 (Kaggle Dataset)
  * **Classes**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mite, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy.
  * **Data Distribution**: The original dataset was modified by reducing the training set and splitting the original validation set to create a dedicated test set.

| Set | Images per Class | Total |
| :--- | :--- | :--- |
| **Training** | 500 | 5000 |
| **Validation**| 50 | 500 |
| **Testing** | 50 | 500 |
| **Total** | **600** | **6000** |

-----

## 📈 Performance & Results

The models were evaluated for accuracy, precision, recall, and F1-score.

### Harumanis Model Results

The Harumanis model achieved an overall **accuracy of 90%**.

| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| Anthracnose | 0.82 | 0.95 | 0.88 |
| Black Sooty Mold | 0.96 | 0.95 | 0.95 |
| Healthy | 0.95 | 0.75 | 0.62 |
| **Weighted Avg** | **0.90** | **0.90** | **0.89** |

### Tomato Model Results

The Tomato model achieved an overall **accuracy of 82%**.

| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| Bacterial Spot | 0.85 | 0.86 | 0.88 |
| Early Blight | 0.80 | 0.64 | 0.71 |
| Late Blight | 0.75 | 0.80 | 0.78 |
| Leaf Mold | 0.95 | 0.80 | 0.70 |
| Septoria Leaf Spot| 0.55 | 0.68 | 0.88 |
| Spider Mite | 0.82 | 0.86 | 0.90 |
| Target Spot | 0.89 | 0.68 | 0.77 |
| Yellow Leaf Curl Virus | 0.90 | 0.88 | 0.89 |
| Mosaic Virus | 0.96 | 0.96 | 0.96 |
| Healthy | 0.98 | 0.90 | 0.94 |
| **Weighted Avg** | **0.84** | **0.82** | **0.82** |

-----

## 🚀 Deployment

The trained models are deployed in two ways for maximum accessibility:

  * **Web Application**: An interactive app built with **Streamlit** that allows users to upload an image or use a camera for instant analysis.
  * **Mobile Application**: A native Android app built using **Kotlin**, which embeds a **TensorFlow Lite (TFLite)** model for efficient on-device processing without needing an internet connection.

-----

## 🛠️ Technologies Used

  * **Modeling**: Python, TensorFlow, Keras
  * **Web Deployment**: Streamlit
  * **Mobile Deployment**: Kotlin, Android Studio, TensorFlow Lite
  * **Development**: Jupyter Notebook

-----

## 💻 Local Setup & Installation

To run the Streamlit web application on your local machine, follow these steps:

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/adzim99/leaf_disease_detection_for_harumanis_tomato.git
    cd leaf_disease_detection_for_harumanis_tomato
    ```

2.  **Create a Virtual Environment (Recommended)**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**

    ```bash
    pip install tensorflow streamlit numpy Pillow
    ```

4.  **Run the Application**

    ```bash
    streamlit run streamlit_app.py
    ```

-----

## 🌱 Future Development

  * **Expand Datasets**: Incorporate more diverse images taken under various conditions to improve model robustness.
  * **Advanced Architectures**: Experiment with more advanced CNN architectures (e.g., ResNet, EfficientNet) to potentially increase accuracy.
  * **Refine Application**: Enhance the user interface and add features like treatment suggestions based on the detected disease.

-----

## 👥 Project Team & Acknowledgements

This project was created for the **WQF7002 Artificial Intelligence Techniques** course (Sem 1, 24/25) at the Faculty of Computer Science & Information Technology, **Universiti Malaya**.

  * MUHAMMAD ADEEB AZIM BIN MOHD NIKMAN (24059750)
  * MUHAMMAD IZWAN FAZRY BIN ABU BAKAR (24064104)
