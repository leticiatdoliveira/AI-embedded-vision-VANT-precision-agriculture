import os
import numpy as np
import tensorflow as tf
from constants import SEED_VALUE, DEFAULT_IMG_SIZE, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA

class Config:
    """Configuration parameters for model training and evaluation."""

    def __init__(self, dataset_type, device='mac', culture=None, balance_classes=True, epochs=DEFAULT_EPOCHS, batch_size = DEFAULT_BATCH_SIZE):
        # Constants from constants.py
        self.IMG_SIZE = DEFAULT_IMG_SIZE
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.LEARNING_RATE = DEFAULT_LEARNING_RATE
        self.DATASET_TYPE = dataset_type
        self.DATA_AUGMENTATION = False
        self.DEVICE = device  # Store device type
        self.BALANCE_CLASSES = balance_classes
    
        # Culture-specific configurations
        if culture:
           self.PLANT_CULTURE = culture

        # Early stopping configuration from constants.py
        self.EARLY_STOP_PATIENCE = EARLY_STOP_PATIENCE
        self.EARLY_STOP_MIN_DELTA = EARLY_STOP_MIN_DELTA

        # Seeds for reproducibility from constants.py
        self.SEED_VALUE = SEED_VALUE
        os.environ['PYTHONHASHSEED'] = str(self.SEED_VALUE)
        tf.random.set_seed(self.SEED_VALUE)
        np.random.seed(self.SEED_VALUE)

        # Path setup
        self.ROOT_PATH = os.getcwd()
        self.PROJECT_PATH = os.path.join(self.ROOT_PATH, self.DATASET_TYPE)
        self.DATASET_PATH = os.path.join(self.PROJECT_PATH, 'dataset')
        self.DATA_PATH = os.path.join(self.PROJECT_PATH, 'data')
        
        # Device-specific directories
        self.MODEL_PATH = os.path.join(self.PROJECT_PATH, 'model', self.DEVICE)
        self.RESULT_PATH = os.path.join(self.PROJECT_PATH, 'results', self.DEVICE)
        self.LOG_PATH = os.path.join(self.PROJECT_PATH, 'log', self.DEVICE)

        # Dataset paths
        self.TRAIN_DIR = os.path.join(self.DATASET_PATH, 'train')
        self.VAL_DIR = os.path.join(self.DATASET_PATH, 'validation')
        self.TEST_DIR = os.path.join(self.DATASET_PATH, 'test')

        # Set base name to save models
        self.MODEL_BASE_NAME = f"ep={self.EPOCHS}_lr={self.LEARNING_RATE}_aug={self.DATA_AUGMENTATION}"

    def create_directories(self):
        """Create the necessary directories if they don't exist."""
        for path in [self.PROJECT_PATH, self.DATASET_PATH, self.MODEL_PATH, self.RESULT_PATH, self.LOG_PATH]:
            if not os.path.exists(path):
                os.makedirs(path)


    def show_directories_path(self):
        print(f"PROJECT PATH: {self.PROJECT_PATH}")
        print(f"DATASET PATH: {self.DATASET_PATH}")
        print(f"MODEL_PATH: {self.MODEL_PATH}")
        print(f"RESULT PATH: {self.RESULT_PATH}")
        print(f"LOG PATH: {self.LOG_PATH}")
        print(f"TRAIN_DIR: {self.TRAIN_DIR}")
        print(f"VAL_DIR: {self.VAL_DIR}")
        print(f"TEST_DIR: {self.TEST_DIR}")
        print(f"MODEL_BASE_NAME: {self.MODEL_BASE_NAME}")
