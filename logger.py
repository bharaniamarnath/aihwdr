import logging

def status(log_file='status.log', level=logging.INFO):
    
    logger = logging.getLogger()
    # Prevent adding handlers multiple times
    if not logger.hasHandlers():
        # Create handlers for console and file
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_file)

        # Set the log level for each handler
        console_handler.setLevel(level)
        file_handler.setLevel(level)

        # Create a formatter and set it for the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        # Set the logging level for the logger
        logger.setLevel(level)
