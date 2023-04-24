import keras
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import os
import time

# Define the alphabet and load the sign language model
alphabet=['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
model = keras.models.load_model("sign_language")

# Function to classify the image
def classify(image):
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    proba=model.predict(image)
    idx = np.argmax(proba)
    return alphabet[idx]

# Initialize the video capture
cap = cv2.VideoCapture(1)
word = ""
prev_alpha = None
last_recognition_time = time.time()

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    # Define the coordinates for the recognition box
    top, right, bottom, left = 100, 750, 400, 1050

    # Extract the region of interest (ROI) and preprocess it
    roi = img[top:bottom, right:left]
    # roi=cv2.flip(roi,1)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Show the preprocessed ROI
    cv2.imshow('roi',gray)

    # Classify the ROI and update the recognized letter if necessary
    alpha=classify(gray)
    if alpha != prev_alpha and time.time() - last_recognition_time > 1:
        prev_alpha = alpha
        last_recognition_time = time.time()

    # Draw the recognition box on the original image
    cv2.rectangle(img, (left, top), (right, bottom), (0,255,0), 2)

    # Display the recognized letter on the image
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, alpha, (0, 130), font, 5, (0, 0, 255), 2)

    # Calculate the position for the recognized word and display it on the image
    text_size, _ = cv2.getTextSize(word, font, 1, 2)
    text_x = (img.shape[1] - text_size[0]) // 2
    cv2.putText(img, word, (text_x, 50), font, 1, (0, 255, 0), 2)

    # Show the final image
    cv2.imshow('img',img)

    # Handle keyboard inputs
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        word += alpha  # Lock the current letter when the space key is pressed

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
