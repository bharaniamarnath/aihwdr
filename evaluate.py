from sklearn.metrics import accuracy_score, classification_report

import logger
import logging
import warnings

logger.status()
warnings.filterwarnings("ignore")

def model_evaluate(models):
    logging.info("Model evaluation started...")
    try:
        # Call model_build to get processed data
        (svm_model, rf_model, keras_model, svm_test_pred, rf_test_pred, keras_test_pred, y_val, y_test) = models

        # Model accuracies
        logging.info("Evaluating models accuracy...")
        acc_svm = accuracy_score(y_val, svm_test_pred)
        acc_rf = accuracy_score(y_val, rf_test_pred)
        acc_keras = accuracy_score(y_test, keras_test_pred)

        # Generate classification reports
        logging.info("Evaluating classification report...")
        report_svm = classification_report(y_val, svm_test_pred)
        report_rf = classification_report(y_val, rf_test_pred)
        report_keras = classification_report(y_test, keras_test_pred)

        # Accuracies summary
        logging.info("Preparing accuracy summary...")
        accuracies = {
            'SVM': {'accuracy': acc_svm, 'report': report_svm},
            'Random Forest': {'accuracy': acc_rf, 'report': report_rf},
            'Keras': {'accuracy': acc_keras, 'report': report_keras}
        }

        logging.info("Completing model evaluation...")
        return accuracies

    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")
        return None
