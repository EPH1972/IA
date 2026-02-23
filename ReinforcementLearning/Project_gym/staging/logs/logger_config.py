import logging
import sys
import os

def get_logger(name):
    """
    Configures and returns a logger instance that writes to both
    console and a file named 'gym_debug.log'.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers if they already exist
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), 'gym_debug.log'))
    
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

if __name__ == "__main__":
    # Test the logger
    log = get_logger("TestLogger")
    log.info("Logger configured successfully.")
    log.debug("This is a debug message (file only).")
