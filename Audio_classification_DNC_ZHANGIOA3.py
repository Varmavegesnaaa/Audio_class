import os
import librosa
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Function to pad or truncate spectrograms to a fixed length
def pad_or_truncate(spectrogram, max_length=128):
    if spectrogram.shape[1] > max_length:
        return spectrogram[:, :max_length]
    else:
        pad_width = max_length - spectrogram.shape[1]
        return np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')

# Function to load audio file
def load_audio(file_path, sr=22050):
    audio, sr = librosa.load(file_path, sr=sr)
    return audio, sr

# Function to extract log Mel-spectrogram
def extract_log_mel_spectrogram(audio, sr, n_mels=128, hop_length=512):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram

# Function to preprocess audio files
def preprocess_audio_files(file_paths, sr=22050, n_mels=128, hop_length=512, max_length=128):
    log_mel_spectrograms = []
    for file_path in file_paths:
        audio, sr = load_audio(file_path, sr)
        log_mel_spectrogram = extract_log_mel_spectrogram(audio, sr, n_mels, hop_length)
        log_mel_spectrogram = pad_or_truncate(log_mel_spectrogram, max_length)
        log_mel_spectrograms.append(log_mel_spectrogram)
    return np.array(log_mel_spectrograms)

# Example usage
audio_folder = 'C:/Audio Classification/DNC'  # Path to the folder containing audio files
file_paths = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith('.wav')]
log_mel_spectrograms = preprocess_audio_files(file_paths)

# Assuming labels are stored in a CSV file
labels_file = 'C:/Audio Classification/extracted_names.csv'
labels_df = pd.read_csv(labels_file)
labels = labels_df['label'].values

# Convert labels to categorical if needed
num_classes = len(np.unique(labels))
labels = to_categorical(labels, num_classes=num_classes)

# Function to build ZhangIOA3 model
def build_zhangioa3_model(input_shape, num_classes=10):
    inputs = Input(shape=input_shape)

    # Block 1
    x = Conv2D(2, (3, 3), padding='valid', strides=(1, 1))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2))(x)

    # Block 2
    x = Conv2D(4, (3, 3), padding='valid', strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2))(x)

    # Block 3
    x = Conv2D(8, (3, 3), padding='valid', strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2))(x)

    # Block 4
    x = Conv2D(16, (3, 3), padding='valid', strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2))(x)

    # Flatten and concatenate input and Block's output
    x = Flatten()(x)
    inputs_flat = Flatten()(inputs)
    x = concatenate([inputs_flat, x])

    # Fully connected layers
    x = Dense(2048)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)

    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)

    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

input_shape = (128, 128, 1)  # Example input shape
model = build_zhangioa3_model(input_shape, num_classes=num_classes)
model.summary()

# Reshape log_mel_spectrograms for input to CNN
log_mel_spectrograms = np.expand_dims(log_mel_spectrograms, axis=-1)  # Add channel dimension

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(log_mel_spectrograms, labels, test_size=0.2, random_state=42)

# Define the checkpoint callback to save the model weights
checkpoint_filepath = 'Audio_class_weights.weights.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# Train the model with the checkpoint callback
model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test),
          callbacks=[model_checkpoint_callback])

# Save the model weights after training
model.save(checkpoint_filepath)