# PRODIGY_ML_04
This code performs the task of training a hand gesture recognition model using TensorFlow and Keras, and then predicting hand gestures from new images. Here's a breakdown of its components:

Libraries: The necessary libraries are imported:

tensorflow.keras for creating deep learning models.
ImageDataGenerator for image preprocessing and augmentation.
LabelEncoder for encoding categorical labels.
matplotlib for plotting images.
skimage.io for reading images.
Data Preprocessing:

ImageDataGenerator is used for loading and augmenting the images from the directory. Images are rescaled (divided by 255) and split into training (80%) and validation sets (20%).
The data directory (data_dir) is where the training images are located. The images are assumed to be organized into subdirectories representing different classes (e.g., different hand gestures).
Model Creation:

A Sequential convolutional neural network (CNN) model is created with:
Three convolutional layers followed by max-pooling layers to extract features from the images.
A dense layer with 512 units and ReLU activation for final classification.
The last dense layer has num_classes neurons (equal to the number of gesture classes) and uses softmax activation for multi-class classification.
The model is compiled using the Adam optimizer, categorical cross-entropy loss, and accuracy as the evaluation metric.
Training:

The model is trained for 10 epochs with the preprocessed training and validation data, and the training progress is tracked in the history object.
After training, the model is saved as hand_gesture_model.h5.
Prediction:

The trained model is loaded from the saved .h5 file.
A function predict_hand_gesture loads a new image, preprocesses it, and predicts the hand gesture.
The predicted gesture is displayed using matplotlib.
Output:

A test image is fed into the model, and the predicted hand gesture ("Ok sign") is printed.
This code demonstrates the full pipeline of data loading, model training, and prediction for hand gesture recognition using deep learning.
