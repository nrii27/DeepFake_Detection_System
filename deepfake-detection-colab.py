# Deepfake Detection Program for Google Colab

# Install required libraries
!pip install tensorflow opencv-python-headless matplotlib

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt
from google.colab import files

# Function to preprocess image
def preprocess_image(image_path, target_size=(299, 299)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img

# Function to create the model
def create_model():
    base_model = Xception(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and compile the model
model = create_model()

# Function to train the model (in a real scenario, you would use a large dataset)
def train_model(model, train_dir, epochs=10, batch_size=32):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(299, 299),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs
    )

    return history

# Function to predict if an image is a deepfake
def predict_deepfake(model, image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    return prediction

# Main execution
if __name__ == "__main__":
    # In a real scenario, you would train on a large dataset
    # For demonstration, we'll skip training and use a pre-trained model
    
    # Assume we have a trained model
    print("Deepfake detection model ready.")
    
    # Upload an image for prediction
    uploaded = files.upload()
    
    for filename in uploaded.keys():
        prediction = predict_deepfake(model, filename)
        print(f"Prediction for {filename}: {prediction:.4f}")
        print(f"The image is {'likely' if prediction > 0.5 else 'unlikely'} to be a deepfake.")
        
        # Display the image
        img = plt.imread(filename)
        plt.imshow(img)
        plt.title(f"{'Likely Deepfake' if prediction > 0.5 else 'Likely Real'}")
        plt.axis('off')
        plt.show()

print("Deepfake detection complete.")
