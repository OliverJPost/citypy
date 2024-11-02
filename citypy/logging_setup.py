import logging
import os
import random
import sys
import time

from loguru import logger

from config import cfg

# Configure a dedicated logger for timing information
log_filename = "function_times.log"
if os.path.exists(log_filename):
    # prevent race condition
    time.sleep(random.random())
    os.remove(log_filename)

# Now add the logger without rotation or retention that might conflict
timing_logger = logger.bind(name="timer")
timing_logger.add(
    log_filename,
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level: <8} | {extra[func_name]: <30} executed in {extra[exec_time]: >8.4f} seconds",
    level="INFO",
    filter=lambda record: record["extra"].get("name") == "timer",
)


def log_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        self = args[0]
        if hasattr(self, "layer_name"):
            func_name = self.layer_name
        else:
            func_name = self.attr_name
        timing_logger.info("Timing", func_name=func_name, exec_time=execution_time)
        return result

    return wrapper


# Configure the logging
# def setup_logging():
#     if not cfg.logging.enable_logging:
#         raise NotImplementedError("Logging can't be disabled at this time")
#
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#
#     # Create a console handler
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.INFO)
#
#     # Create a file handler
#     file_handler = logging.FileHandler(cfg.logging.log_filename)
#     file_handler.setLevel(logging.INFO)
#
#     # Define a log format
#     formatter = logging.Formatter(cfg.logging.message_format)
#     console_handler.setFormatter(formatter)
#     file_handler.setFormatter(formatter)
#
#     # Add handlers to the logger
#     if not logger.hasHandlers():  # Avoid adding multiple handlers
#         logger.addHandler(console_handler)
#         logger.addHandler(file_handler)
#
#     return logger
#
#
# # Initialize logger
# logger = setup_logging()
#
#
# def handle_exception(exc_type, exc_value, exc_traceback):
#     if issubclass(exc_type, KeyboardInterrupt):
#         sys.__excepthook__(exc_type, exc_value, exc_traceback)
#         return
#
#     logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
#
#
# sys.excepthook = handle_exception
