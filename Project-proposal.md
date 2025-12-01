# Project Proposal: Deep Learning for Facial Expression Recognition in Uncontrolled Environments

**Team Name:** [Insert Team Name Here]
**Team Members:** [Member 1], [Member 2], [Member 3]
**Course:** COMPSCI 178 – Machine Learning and Data Mining

## 1. Problem Statement

Facial Expression Recognition (FER) is a critical component of human-computer interaction, used in applications ranging from analyzing customer satisfaction to monitoring driver alertness. While FER systems perform well on posed, studio-quality images (e.g., CK+ dataset), their performance degrades significantly in "in-the-wild" scenarios where lighting, pose, and occlusion vary unpredictably.

Our project aims to build a robust emotion classification system capable of categorizing faces into standard emotion categories (e.g., Happiness, Anger, Sadness, Neutral) using the **Muxspace Facial Expression** dataset. We will investigate whether deep learning techniques can effectively overcome the noise inherent in uncontrolled environments compared to traditional feature extraction methods.

## 2. Dataset Analysis

We will use the **Muxspace Facial Expression** dataset, which consists of approximately 13,000 images of facial expressions.

* **Characteristics:** The images represent real-world conditions rather than controlled lab settings.
* **Classes:** The data is labeled into 7 distinct emotion categories.
* **Challenges:** Preliminary analysis suggests significant variance in head pose, lighting conditions, and potential background clutter. The dataset size (~13k) is relatively small for deep learning, posing a risk of overfitting that our methodology must address.

## 3. Proposed Methodology

To ensure a rigorous comparative evaluation, we will implement a pipeline that progresses from traditional baselines to advanced deep learning architectures.

### A. Data Preprocessing & Augmentation

Since the dataset contains background noise, raw images cannot be fed directly into classifiers.

* **Face Detection:** We will employ a Haar Cascade or MTCNN detector to crop faces from the raw images, removing irrelevant background information.
* **Normalization:** Images will be resized to a fixed resolution (e.g., 48x48 or 224x224) and pixel values normalized.
* **Augmentation:** To combat the limited dataset size, we will apply random horizontal flips, slight rotations, and zoom shifts during training to improve model generalization.

### B. Baseline Model (Traditional ML)

We will establish a performance baseline using "hand-crafted" features:

* **Feature Extraction:** Histogram of Oriented Gradients (HOG) to capture shape and texture information.
* **Classifier:** Support Vector Machine (SVM) or Random Forest.
* **Goal:** This will serve as the lower bound for accuracy, quantifying how much performance is gained by moving to learned representations.

### C. Primary Model (Custom CNN)

We will design and train a Convolutional Neural Network (CNN) from scratch.

* **Architecture:** A standard architecture consisting of 3-4 convolutional blocks (Conv2D + ReLU + Batch Normalization + MaxPool) followed by dense layers.
* **Focus:** Optimizing dropout rates and regularization weight decay to prevent overfitting on the 13k images.

## 4. Advanced Techniques ("Something Extra")

To satisfy the project requirements for depth and breadth, we will implement two advanced components:

### A. Transfer Learning (Breadth)

Given the dataset size, training from scratch may result in suboptimal feature learning. We will implement **Transfer Learning** using a state-of-the-art architecture (e.g., **ResNet-18** or  **VGG16** ) pre-trained on ImageNet.

* **Strategy:** We will freeze the early feature-extraction layers and fine-tune the final fully connected layers for our 7-class emotion problem.
* **Hypothesis:** We expect this model to significantly outperform both the baseline and our custom CNN by leveraging features learned from millions of images.

### B. Model Interpretability with Grad-CAM (Depth)

Deep learning models are often criticized as "black boxes." To address this, we will implement  **Gradient-weighted Class Activation Mapping (Grad-CAM)** .

* **Objective:** We will generate heatmaps for test images to visualize exactly *where* the CNN is looking when it predicts an emotion.
* **Analysis:** We will analyze if the model focuses on semantically relevant features (e.g., the mouth for "Happiness", the eyes/brows for "Anger") or if it is cheating by looking at background artifacts.

## 5. Evaluation Strategy

We will not rely on a single train/test split.

* **Metrics:** Accuracy, Precision, Recall, and F1-Score per class.
* **Validation:** We will use k-fold cross-validation (e.g., k=5) to ensure our results are statistically robust.
* **Error Analysis:** We will produce Confusion Matrices to specifically analyze common misclassifications (e.g., distinguishing "Fear" from "Surprise") and discuss why they occur.foll
