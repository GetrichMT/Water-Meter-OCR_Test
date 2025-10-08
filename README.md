# Water-Meter-OCR_Test

This project automatically reads water meter digits from images using a custom-trained Optical Character Recognition (OCR) model.
It uses TensorFlow and OpenCV to train a neural network that recognizes the numeric readings displayed on analog or digital water meters.

train.py (Model Training)
Trains a convolutional + recurrent neural network (CNN+RNN) to recognize digits on the water meter display using labeled images from labels.csv.

model.h5 (Trained Model)
Contains the trained neural network weights and architecture (the “brain”). This file is created after training and can be reused for predictions.

inference.py (Prediction)
Loads model.h5 and predicts the numeric value from a new meter image. This is the main file used for testing or deployment.

preprocess.py (Image Processing)
Converts meter images into standardized grayscale format, resizes them, and normalizes pixel values before feeding them to the model.

labels.csv (data)
A list of all training image filenames and their corresponding true readings (e.g., "003846.jpg,3846"). This acts as the answer key during model training.

images 
Sample image for training
