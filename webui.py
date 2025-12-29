import gradio as gr
from output import predict_digit

import logger
import logging
import warnings

logger.status()
warnings.filterwarnings("ignore")

def create_sample_digits_iface(outputs):
    logging.info("Building sample digits interface...")
    return gr.Interface(
        fn=lambda: outputs[0]['sample_digits'], 
        inputs=None,
        outputs="image",
        title="MNIST Sample Digits",
        description="View the sample digits in the MNIST training dataset."
    )

def create_data_distribution_iface(outputs):
    logging.info("Building data disttribution interface...")
    return gr.Interface(
        fn=lambda: outputs[0]['data_distribution'],
        inputs=None,
        outputs="image",
        title="MNIST Data Distribution",
        description="View the distribution of digits in the MNIST training dataset."
    )

def create_data_correlation_iface(outputs):
    logging.info("Building data correlation interface...")
    return gr.Interface(
        fn=lambda: outputs[0]['data_correlation'],
        inputs=None,
        outputs="image",
        title="MNIST Data Correlation",
        description="View the data correlation in the MNIST training dataset."
    )

def create_models_comparison_iface(outputs):
    logging.info("Building models comparison interface...")
    return gr.Interface(
        fn=lambda: outputs[0]['models_comparison'],
        inputs=None,
        outputs="image",
        title="MNIST Models Comparison",
        description="View the models comparison in the MNIST digit prediction."
    )

def create_prediction_iface(models):
    logging.info("Building model prediction interface...")
    if models is None or len(models) < 3:
        raise ValueError("Expected model outputs to be a tuple of (svm_model, rf_model, keras_model)")
    return gr.Interface(
        fn=lambda image: (
            predict_digit(models, image) if image is not None else {"error": "No image provided"}
        ),
        inputs=gr.Image(shape=(28, 28), type='numpy'),
        outputs="json",
        title="Handwritten Digit Recognition",
        description="Upload a handwritten digit (0-9) image. Preferred format: JPG/PNG with a 1:1 ratio."
    )

def web_interface(outputs, models):
    logging.info("Web UI building started...")
    gr.TabbedInterface(
        [create_sample_digits_iface(outputs), 
         create_data_distribution_iface(outputs), 
         create_data_correlation_iface(outputs), 
         create_models_comparison_iface(outputs), 
         create_prediction_iface(models)], 
        ["Sample Digits", "Data Distribution", "Data Correlation", "Models Comparison", "Prediction"]
    ).launch()