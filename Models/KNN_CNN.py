import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Load the CSV files
features_mean = pd.read_csv('archive/Data/features_30_sec.csv')
features_split = pd.read_csv('archive/Data/features_3_sec.csv')

# Load the audio files from 'genres_original' and ensure alignment with labels
audio_files = []
audio_labels = []
for genre in features_mean['label']:
    for i in range(100):
        filename = f'archive/Data/genres_original/{genre}/{genre}.{str(i).zfill(5)}.wav'
        audio_files.append(filename)
        audio_labels.append(genre)

# Load the images from 'images_original' and ensure alignment with audio data
image_files = []
for genre in features_mean['label']:
    for i in range(100):
        if not (genre == 'jazz' and i == 29):  # Handle the missing image
            filename = f'archive/Data/images_original/{genre}/{genre}{str(i).zfill(5)}.png'
            image_files.append(filename)

# Initialize the label encoder and fit it to the labels
label_encoder = LabelEncoder()
label_encoder.fit(audio_labels)

def preprocess_data(df, scaler, batch_size=None):
    # Extract features and labels
    X_audio = df.drop(['label', 'filename'], axis=1)  # Audio features

    if batch_size:
        X_images = load_and_preprocess_images_in_batches(image_files, batch_size)  # Load and preprocess images
    else:
        X_images = load_and_preprocess_images(image_files)  # Load and preprocess images

    y = label_encoder.transform(df['label'])  # Encoded labels

    # Split the data into training and testing sets
    X_audio_train, X_audio_test, X_images_train, X_images_test, y_train, y_test = train_test_split(
        X_audio, X_images, y, test_size=0.2, random_state=42)

    # Standardize the audio features
    X_audio_train = scaler.fit_transform(X_audio_train)
    X_audio_test = scaler.transform(X_audio_test)

    return X_audio_train, X_audio_test, X_images_train, X_images_test, y_train, y_test


# Function to load and preprocess images in batches
def load_and_preprocess_images_in_batches(image_files, batch_size):
    images = []
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_images = []
        for file in batch_files:
            if os.path.exists(file):  # Check if the image file exists
                img = load_img(file, target_size=(224, 224))
                img = img_to_array(img)
                img = preprocess_input(img)
                batch_images.append(img)
        images.extend(batch_images)
    return np.array(images)

# Function to load and preprocess images with error handling
def load_and_preprocess_images(image_files):
    images = []
    for file in image_files:
        if os.path.exists(file):  # Check if the image file exists
            img = load_img(file, target_size=(224, 224))
            img = img_to_array(img)
            img = preprocess_input(img)
            images.append(img)
    return np.array(images)

# Specify the batch size
batch_size = 32  # Adjust as needed

# Preprocess 'features_mean' dataset without batch loading
X_audio_mean_train, X_audio_mean_test, X_images_mean_train, X_images_mean_test, y_mean_train, y_mean_test = preprocess_data(features_mean, StandardScaler())

# Preprocess 'features_split' dataset with batch loading
X_audio_split_train, X_audio_split_test, X_images_split_train, X_images_split_test, y_split_train, y_split_test = preprocess_data(features_split, StandardScaler(), batch_size)

# Define image dimensions and number of classes
image_height, image_width, image_channels = 224, 224, 3
num_classes = len(label_encoder.classes_)

# Implement a neural network that combines audio and image data for classification
def create_combined_model(input_shape_audio, input_shape_images, num_classes):
    # Define the audio branch of the model
    audio_input = Input(shape=input_shape_audio)
    audio_dense = Dense(64, activation='relu')(audio_input)

    # Define the image branch of the model
    image_input = Input(shape=input_shape_images)
    image_conv1 = Conv2D(32, (3, 3), activation='relu')(image_input)
    image_maxpool = MaxPooling2D((2, 2))(image_conv1)
    image_conv2 = Conv2D(64, (3, 3), activation='relu')(image_maxpool)
    image_flatten = Flatten()(image_conv2)

    # Combine the audio and image branches
    combined = concatenate([audio_dense, image_flatten])

    # Add fully connected layers for classification
    x = Dense(128, activation='relu')(combined)
    x = Dense(64, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[audio_input, image_input], outputs=output)
    return model

# Create and compile the combined model for 'features_mean' dataset
combined_model_mean = create_combined_model(input_shape_audio=X_audio_mean_train.shape[1],
                                            input_shape_images=(image_height, image_width, image_channels),
                                            num_classes=num_classes)
combined_model_mean.compile(optimizer=Adam(learning_rate=0.001),
                            loss='tf.compat.v1.losses.sparse_softmax_cross_entropy',
                            metrics=['accuracy'])

# Train the model on GPU
#with tf.device('/GPU:0'):
#    combined_model_mean.fit([X_audio_mean_train, X_images_mean_train], y_mean_train, epochs=10, batch_size=32)
history = combined_model_mean.fit([X_audio_mean_train, X_images_mean_train], y_mean_train,
                                  epochs=10, batch_size=32,
                                  validation_data=([X_audio_mean_test, X_images_mean_test], y_mean_test),
                                  callbacks=[checkpoint])

for epoch, acc in enumerate(history.history['val_accuracy']):
    print(f"Epoch {epoch + 1}: Validation Accuracy = {acc:.4f}")
    
# Evaluate the model
y_mean_pred = combined_model_mean.predict([X_audio_mean_test, X_images_mean_test])
accuracy_mean = accuracy_score(y_mean_test, y_mean_pred.argmax(axis=-1))
report_mean = classification_report(y_mean_test, y_mean_pred.argmax(axis=-1))
confusion_mean = confusion_matrix(y_mean_test, y_mean_pred.argmax(axis=-1))

# Print evaluation results for 'features_mean' dataset
print("Accuracy (Mean Features):", accuracy_mean)
print("Classification Report (Mean Features):\n", report_mean)
print("Confusion Matrix (Mean Features):\n", confusion_mean)