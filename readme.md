
## Real-Time American Sign Language (ASL) Recognition System
This project aims to develop a real-time American Sign Language (ASL) recognition system using Convolutional Neural Networks (CNNs) to improve communication and accessibility for the hearing-impaired community. By implementing this system in communication aids, educational resources, and assistive technologies, we hope to bridge the communication gap and enhance the quality of life for individuals who rely on ASL for communication.

## Demo 

https://user-images.githubusercontent.com/61319491/234713490-3e90016d-eef1-4c40-b996-27f8e259514e.mp4


## Overview
The project involves preprocessing input images using techniques such as resizing, grayscaling, and Gaussian blurring, and training a CNN model with multiple layers, including convolutional, max-pooling, dropout, and dense layers. The trained model is integrated into a real-time video stream using OpenCV for ASL gesture recognition, and the chosen model architecture is designed to optimize performance while minimizing overfitting.

Utilizing the Sign Language MNIST dataset, designed for more challenging image-based machine learning benchmarks in real-world applications, the project applies preprocessing steps, such as resizing, grayscaling, and Gaussian blurring, to the input images to improve the performance of the CNN model. The images are then converted to arrays and normalized before being fed into the model. The developed model demonstrates high training and validation accuracy, showcasing the effectiveness of the proposed approach in recognizing ASL gestures in real-time.

Future work includes expanding the range of recognized ASL signs, improving model robustness to variations in lighting, hand orientation, and background conditions, and exploring techniques to increase system efficiency and reduce processing time for real-time implementation.

## Repository Contents
- Prediction.py: Script for real-time ASL gesture recognition using the trained CNN model and OpenCV.
- Training.ipynb: Jupyter Notebook containing the preprocessing steps, CNN model architecture, and training process.


## Getting Started
- Clone the repository:

```
git clone https://github.com/your_username/ASL-Recognition.git
```

- Install the required libraries:

```
pip install -r requirements.txt
```

- Run the Jupyter Notebook to train the model:
```
jupyter notebook Training.ipynb
```
- Run the Prediction.py script to start real-time ASL gesture recognition:
```
python Prediction.py
```
