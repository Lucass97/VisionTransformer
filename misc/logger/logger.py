import logging
from misc.decorators import singleton


@singleton
class CustomLogger:
    COLOR_INFO = '\033[32m'      # Green for INFO
    COLOR_WARNING = '\033[33m'   # Yellow for WARNING
    COLOR_ERROR = '\033[31m'     # Red for ERROR
    COLOR_DEBUG = '\033[36m'     # Cyan for DEBUG
    COLOR_CRITICAL = '\033[35m'  # Magenta for CRITICAL
    COLOR_RESET = '\033[0m'      # Reset color

    def __init__(self) -> None:
        """
        Initializes the custom logger with color-coded log levels.

        Sets up the logger to print messages with specific colors for different log levels.
        """
        self.logger = logging.getLogger("CustomLogger")
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

    def _colorize(self, message, color_code) -> str:
        """
        Adds color to the log message.

        Args:
            message (str): The log message.
            color_code (str): The color code to apply.

        Returns:
            str: The colored log message.
        """
        return f"{color_code}{message}{self.COLOR_RESET}"

    def info(self, message: str) -> None:
        """Logs an info message with green color."""
        self.logger.info(self._colorize(message, self.COLOR_INFO))

    def warning(self, message: str) -> None:
        """Logs a warning message with yellow color."""
        self.logger.warning(self._colorize(message, self.COLOR_WARNING))

    def error(self, message: str) -> None:
        """Logs an error message with red color."""
        self.logger.error(self._colorize(message, self.COLOR_ERROR))

    def debug(self, message: str) -> None:
        """Logs a debug message with cyan color."""
        self.logger.debug(self._colorize(message, self.COLOR_DEBUG))

    def critical(self, message: str) -> None:
        """Logs a critical message with magenta color."""
        self.logger.critical(self._colorize(message, self.COLOR_CRITICAL))
        