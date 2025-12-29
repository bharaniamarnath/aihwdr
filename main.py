# PTID-CDS-DEC-25-3624_PRCP-1002-handwritten-digits-recognition
import logger
import logging
import warnings

from process import data_process
from model import model_build
from evaluate import model_accuracy
from output import model_output
from webui import web_interface

logger.status()
warnings.filterwarnings("ignore")

def main():
    try:
        # Data Processing
        logging.info("Begin data processing...")
        data = data_process()
        if data is None:
            raise ValueError("Data processing failed.")

        # Model Building
        logging.info("Begin model building...")
        models = model_build(data)
        if models is None:
            raise ValueError("Model building failed.")

        # Model Evaluation
        logging.info("Begin model evaluation...")
        accuracies = model_accuracy(models)
        if accuracies is None:
            raise ValueError("Model evaluation failed.")

        # Output Generation
        logging.info("Begin output generation...")
        outputs = model_output(data, models, accuracies)
        if outputs is None:
            raise ValueError("Output generation failed.")

        # Launch Web Interface
        logging.info("Begin web UI building...")
        web_interface(outputs, models)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
