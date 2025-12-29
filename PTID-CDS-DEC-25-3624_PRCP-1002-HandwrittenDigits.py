# PTID-CDS-DEC-25-3624_PRCP-1002-handwritten-digits-recognition

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import gradio as gr
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the dataset
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Split training data for validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# Scale data for SVM and Random Forest
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_val_flat = x_val.reshape(x_val.shape[0], -1)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_flat)
x_val_scaled = scaler.transform(x_val_flat)

# Train SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(x_train_scaled, y_train)

# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train_flat, y_train)

# Build and compile Keras model
keras_model = Sequential()
keras_model.add(Flatten(input_shape=(28, 28)))
keras_model.add(Dense(128, activation='relu'))
keras_model.add(Dense(10, activation='softmax'))

# Train Keras model using original images
keras_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
keras_model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

# Model prediction
svm_test_pred = svm_model.predict(x_val_scaled)
rf_test_pred = rf_model.predict(x_val_scaled)
keras_test_pred = np.argmax(keras_model.predict(x_test.reshape(-1, 28, 28)), axis=1)

# Model accuracies
# Accuracy report
acc_svm = accuracy_score(y_val, svm_test_pred)
acc_rf = accuracy_score(y_val, rf_test_pred)
acc_keras = accuracy_score(y_test, keras_test_pred)

# Calculate accuracy for each model
accuracies = {
    'SVM': acc_svm,
    'Random Forest': acc_rf,
    'Keras': acc_keras
}

# Output functions

# Display Samples Images
def show_sample_digits():
    plt.figure(figsize=(15, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_train[i], cmap='gray')
        plt.title(f'Label: {y_train[i]}')
        plt.axis('off')
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('digit_samples.png')
    # Close the figure to avoid display
    plt.close()
    return 'digit_samples.png'

# Display digit distribution function
def show_data_distribution():
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y_train, palette='tab20')
    plt.title('MNIST digits distribution in training data')
    plt.xlabel('Digit')
    plt.ylabel('Frequency')
    plt.xticks(ticks=np.arange(10), labels=np.arange(10))
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig('data_distribution.png')
    # Close the figure to avoid display
    plt.close()
    return 'data_distribution.png'

# Display data correlation function
def show_data_correlation():
    # Correlation from 10k training data images
    sample_x = x_train.reshape(x_train.shape[0], 28 * 28)
    sample_df = pd.DataFrame(sample_x[:10000])

    # Correlation matrix
    correlation_matrix = sample_df.corr()

    # CM plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('MNIST sample correlation matrix')

    # Save the plot to a file
    plt.savefig('data_correlation.png')
    # Close the figure to avoid display
    plt.close()
    return 'data_correlation.png'

# Display models comparison
def show_models_comparison():
    models = list(accuracies.keys())
    plt.bar(models, accuracies.values())
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim([0, 1])

    # Save the plot to a file
    plt.savefig('models_comparison.png')
    # Close the figure to avoid display
    plt.close()
    return 'models_comparison.png'

# Predict digit from image function
def predict_digit(image):
    # Convert array back to image
    img = Image.fromarray(image.astype(np.uint8))
    # Convert to grayscale
    img = img.convert('L')
    # Resize to 28x28
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32)
    # Invert colors
    img_array = 255 - img_array
    # Normalize
    img_array = img_array / 255.0
    
    # Flatten for SVM and Random Forest
    img_flattened = img_array.flatten().reshape(1, -1)

    # Predictions from all models
    svm_prediction = svm_model.predict(scaler.transform(img_flattened))
    rf_prediction = rf_model.predict(img_flattened)
    keras_prediction = np.argmax(keras_model.predict(img_array.reshape(1, 28, 28)), axis=1)

    return {
        "SVM Prediction": int(svm_prediction[0]),
        "Random Forest Prediction": int(rf_prediction[0]),
        "Keras Prediction": int(keras_prediction[0])
    }

# Gradio Section

# Gradio Interface for Sample Digits
sample_digits_iface = gr.Interface(
    fn=show_sample_digits,
    inputs=None,  # No input needed for this function
    outputs="image",
    title="MNIST Sample Digits",
    description="View the sample digits in the MNIST training dataset."
    )

# Gradio Interface for Data Distribution
data_distribution_iface = gr.Interface(
    fn=show_data_distribution,
    inputs=None,  # No input needed for this function
    outputs="image",
    title="MNIST Data Distribution",
    description="View the distribution of digits in the MNIST training dataset."
    )

# Gradio Interface for Data Correlation
data_correlation_iface = gr.Interface(
    fn=show_data_correlation,
    inputs=None,  # No input needed for this function
    outputs="image",
    title="MNIST Data Correlation",
    description="View the data correlation in the MNIST training dataset."
    )

# Gradio Interface for Models Comparison
models_comparison_iface = gr.Interface(
    fn=show_models_comparison,
    inputs=None,  # No input needed for this function
    outputs="image",
    title="MNIST Models Comparison",
    description="View the models comparison in the MNIST digit prediction."
    )

# Gradio interface for Prediction
prediction_iface = gr.Interface(
    fn=predict_digit, 
    inputs=gr.inputs.Image(shape=(28, 28)), 
    outputs=["json"],
    title="Handwritten Digit Recognition: Using SVM, Random Forest, Keras models",
    description="Upload handwritten digit (0-9) image. Preferred format: JPG/PNG 1:1 ratio."
    )

# Launch both interfaces
# data_distribution_iface.launch()
# prediction_iface.launch()

gr.TabbedInterface(
    [sample_digits_iface, data_distribution_iface, data_correlation_iface, models_comparison_iface, prediction_iface], 
    ["Sample Digits", "Data Distribution", "Data Correlation", "Models Comparison", "Prediction"]
).launch()

