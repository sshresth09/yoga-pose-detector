# Yoga-pose-detector

## About
This system leverages deep learning to identify yoga poses from live video feeds. It consists of two components: yoga-detection.py for real-time detection and model_training.py for training the pose classification model.

## Approach
The project utilizes a combination of YOLOv8 for object detection and a custom CNN model for yoga pose classification. YOLOv8 detects persons in video frames, while the CNN model predicts the specific yoga pose of the cropped person.

## Data
Dataset Source Link: [yoga pose images](https://universe.roboflow.com/new-workspace-mujgg/yoga-pose)



## Data Preprocessing
### a. Dataset
Images for yoga poses are structured into training (data/train) and validation (data/valid) directories.
Each class corresponds to a yoga pose, with 107 unique labels derived from traditional yoga poses.
### b. Image Augmentation
Rescaling: All images are normalized to have pixel values between 0 and 1.
Target Size: Images are resized to 64x64 to match the input dimensions of the CNN model.
### c. Batch Processing
ImageDataGenerator is used to preprocess and feed the data to the model in batches

## Architecture
### a. CNN Classifier
<li>Input Layer: Accepts 64x64 RGB images.

<li>Convolutional Layers: Three layers with ReLU activation and batch normalization for feature extraction.

<li>Pooling Layers: Max pooling with a stride of 2 to downsample feature maps.

<li>Fully Connected Layers:

<li>Dense layer with 512 neurons for feature aggregation.

<li>Dropout of 0.5 to prevent overfitting.

<li>Dense layer with 256 neurons, followed by another dropout.

<li>Output layer with 107 neurons using a softmax activation function for pose classification.
  
### b. Training
<li>Loss Function: Sparse Categorical Crossentropy.
<li>Optimizer: Adam optimizer for adaptive learning.
<li>Metrics: Accuracy on training and validation data.
<li>Epochs: 20 training iterations.

## Setup

1. Download and unzip the dataset.

2. Clone the repository
   
``` git clone https://github.com/sshresth09/yoga-pose-detector.git```

3. Install dependencies
   
``` pip install -r requirements.txt ```

4. Run the Python File

```python yoga-detection.py```
