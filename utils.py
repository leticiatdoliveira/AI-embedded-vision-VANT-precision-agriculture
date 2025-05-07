import tensorflow as tf

class Utils:
    """Utility functions for the project."""

    @staticmethod
    def check_gpu():
        """Check for GPU availability."""
        print(f"TensorFlow version: {tf.__version__}")
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        print("List of GPUs Available: ", tf.config.list_physical_devices('GPU'))
