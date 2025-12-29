import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import StandardScaler

import logger
import logging
import warnings

logger.status()
warnings.filterwarnings("ignore")

# Display Sample Images
def show_sample_digits(x_train, y_train):
    plt.figure(figsize=(15, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_train[i], cmap='gray')
        plt.title(f'Label: {y_train[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('digit_samples.png')
    plt.close()
    return 'digit_samples.png'

# Display digit distribution function
def show_data_distribution(y_train):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y_train, palette='tab20')
    plt.title('MNIST Digits Distribution in Training Data')
    plt.xlabel('Digit')
    plt.ylabel('Frequency')
    plt.xticks(ticks=np.arange(10), labels=np.arange(10))
    plt.tight_layout()
    plt.savefig('data_distribution.png')
    plt.close()
    return 'data_distribution.png'

# Display data correlation function
def show_data_correlation(x_train):
    sample_x = x_train.reshape(x_train.shape[0], 28 * 28)
    sample_df = pd.DataFrame(sample_x[:10000])
    correlation_matrix = sample_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('MNIST Sample Correlation Matrix')
    plt.savefig('data_correlation.png')
    plt.close()
    return 'data_correlation.png'

# Display models comparison
def show_models_comparison(accuracies):
    models = list(accuracies.keys())
    accuracy_values = [accuracies[model]['accuracy'] for model in models]
    plt.bar(models, accuracy_values)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim([0, 1])
    plt.savefig('models_comparison.png')
    plt.close()
    return 'models_comparison.png'

# Predict digit from image function
# def predict_digit(svm_model, rf_model, keras_model, image_to_predict):
def predict_digit(models, image_to_predict):
    (svm_model, rf_model, keras_model, _, _, _, _, _) = models
    # Convert array back to image
    img = Image.fromarray(image_to_predict.astype(np.uint8)).convert('L').resize((28, 28))
    img_array = np.array(img, dtype=np.float32)
    img_array = 255 - img_array  # Invert colors for MNIST
    img_array /= 255.0  # Normalize
    
    # Flatten for SVM and Random Forest
    img_flattened = img_array.flatten().reshape(1, -1)

    # Use the scaler from the original data processing
    svm_prediction = svm_model.predict(img_flattened)
    rf_prediction = rf_model.predict(img_flattened)
    keras_prediction = np.argmax(keras_model.predict(img_array.reshape(1, 28, 28)), axis=1)

    return {
        "SVM Prediction": int(svm_prediction[0]),
        "Random Forest Prediction": int(rf_prediction[0]),
        "Keras Prediction": int(keras_prediction[0])
    }

# Main function to output results
def model_output(data, models, accuracies, image_to_predict=None):
    logging.info("Output generation started...")
    (x_train, y_train, _, _, _, _, _, _, _, _, _) = data
    # (svm_model, rf_model, keras_model, _, _, _, _, _) = models
    accuracies = accuracies
    output_images = {
        "sample_digits": show_sample_digits(x_train, y_train),
        "data_distribution": show_data_distribution(y_train),
        "data_correlation": show_data_correlation(x_train),
        "models_comparison": show_models_comparison(accuracies)
    }
    
    prediction_result = None
    if image_to_predict is not None:
        # prediction_result = predict_digit(svm_model, rf_model, keras_model, image_to_predict)
        prediction_result = predict_digit(models, image_to_predict)

    logging.info("Completing output generation...")
    return output_images, prediction_result


