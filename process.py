import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import logger
import logging
import warnings

logger.status()
warnings.filterwarnings("ignore")

def data_process():
    logging.info("Data processing started...")
    try:
        # Load MNIST dataset
        logging.info("Loading dataset...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Normalize the dataset
        logging.info("Normalizing data...")
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # Split training data for validation
        logging.info("Splitting data...")
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

        # Scale data for SVM and Random Forest
        logging.info("Scaling data...")
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_val_flat = x_val.reshape(x_val.shape[0], -1)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_flat)
        x_val_scaled = scaler.transform(x_val_flat)

        logging.info("Completing data processing...")
        return (x_train, y_train, x_test, y_test, x_val, y_val, x_train_flat, x_val_flat, x_train_scaled, x_val_scaled, scaler)

    except Exception as e:
        print(f"An error occurred during data processing: {e}")
        return None
