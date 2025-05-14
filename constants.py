"""Constants used throughout the project."""

# Model training constants
DEFAULT_BATCH_SIZE = 8
DEFAULT_IMG_SIZE = (224, 224)
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 10

# Early stopping configuration
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 0.001

# Seed for reproducibility
SEED_VALUE = 42

# File extensions
KERAS_EXTENSION = ".keras"
H5_EXTENSION = ".h5"
TFLITE_EXTENSION = ".tflite"

# Dataset split ratios
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO =  1.0 - (TRAIN_RATIO + VALIDATION_RATIO)
