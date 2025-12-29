import numpy as np
from process import data_process
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

import logger
import logging
import warnings

logger.status()
warnings.filterwarnings("ignore")

def model_build(data):
    logging.info("Model building started...")
    try:
        # Load processed data
        logging.info("Loading processed data...")
        (x_train, y_train, x_test, y_test, x_val, y_val, x_train_flat, x_val_flat, x_train_scaled, x_val_scaled, scaler) = data

        # Train SVM model
        logging.info("Building SVM model...")
        svm_model = SVC(kernel='linear', random_state=42)
        svm_model.fit(x_train_scaled, y_train)

        # Train Random Forest model
        logging.info("Building RF model..")
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(x_train_flat, y_train)

        # Build and compile Keras model
        logging.info("Building KS model...")
        keras_model = Sequential()
        keras_model.add(Flatten(input_shape=(28, 28)))
        keras_model.add(Dense(128, activation='relu'))
        keras_model.add(Dense(10, activation='softmax'))

        # Compile the Keras model
        logging.info("Compiling KS model...")
        keras_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        keras_model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

        # Model predictions
        logging.info("Models prediction with test values...")
        svm_test_pred = svm_model.predict(x_val_scaled)
        rf_test_pred = rf_model.predict(x_val_flat)
        keras_test_pred = np.argmax(keras_model.predict(x_test.reshape(-1, 28, 28)), axis=1)

        logging.info("Completing model building...")
        return (svm_model, rf_model, keras_model, svm_test_pred, rf_test_pred, keras_test_pred, y_val, y_test)

    except Exception as e:
        print(f"An error occurred during model building: {e}")
        return None  # Return None if an error occurs
