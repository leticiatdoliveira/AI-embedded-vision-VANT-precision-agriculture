import os
import logging


class Logger:
    """Handles logging configuration and operations."""

    def __init__(self, output_dir, file_name):
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False
        self.configure_handlers(output_dir, file_name)

    def configure_handlers(self, output_dir, file_name):
        filename = file_name + '.log'

        # Check if handlers are already added
        if not self.logger.handlers:
            # Console handler
            console_formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
            console_handler = logging.StreamHandler()
            self.configure_logger(logging.INFO, console_formatter, console_handler)

            # File handler
            file_formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
            file_handler = logging.FileHandler(os.path.join(output_dir, filename), "w")
            self.configure_logger(logging.DEBUG, file_formatter, file_handler)

    def configure_logger(self, level, formatter, handler):
        handler.setLevel(level)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(level)  # Set overall logger level

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

    def critical(self, message):
        self.logger.critical(message)
