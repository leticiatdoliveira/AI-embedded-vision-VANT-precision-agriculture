import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot
import matplotlib.pyplot as plt
import time
import tracemalloc
from logger import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all messages, 1 = INFO, 2 = WARNING, 3 = ERROR


class Config:
    """Configuration parameters for model training and evaluation."""

    def __init__(self, dataset_type):
        # Constants
        self.IMG_SIZE = (224, 224)
        self.BATCH_SIZE = 32
        self.EPOCHS = 10
        self.LEARNING_RATE = 0.001
        self.DATASET_TYPE = dataset_type
        self.DATA_AUGMENTATION = False

        # Early stopping configuration
        self.EARLY_STOP_PATIENCE = 5
        self.EARLY_STOP_MIN_DELTA = 0.001

        # Seeds for reproducibility
        self.SEED_VALUE = 42
        os.environ['PYTHONHASHSEED'] = str(self.SEED_VALUE)
        tf.random.set_seed(self.SEED_VALUE)
        np.random.seed(self.SEED_VALUE)

        # Path setup
        self.ROOT_PATH = os.getcwd()
        self.PROJECT_PATH = os.path.join(self.ROOT_PATH, self.DATASET_TYPE)
        self.DATASET_PATH = os.path.join(self.PROJECT_PATH, 'dataset')
        self.MODEL_PATH = os.path.join(self.PROJECT_PATH, 'model')
        self.RESULT_PATH = os.path.join(self.PROJECT_PATH, 'results')
        self.LOG_PATH = os.path.join(self.PROJECT_PATH, 'log')

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


class DataLoader:
    """Handles dataset creation and processing."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def create_tf_dataset(self, dataset_path, label_type='inferred', label_mode='categorical', shuffle=True):
        """Create a TensorFlow dataset from directory."""
        return tf.keras.utils.image_dataset_from_directory(
            dataset_path,
            labels=label_type,
            label_mode=label_mode,
            image_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE,
            shuffle=shuffle,
            seed=self.config.SEED_VALUE
        )

    @staticmethod
    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    def load_datasets(self):
        """Load and prepare all datasets."""
        # Create dataset objects
        train_ds = self.create_tf_dataset(self.config.TRAIN_DIR)
        val_ds = self.create_tf_dataset(self.config.VAL_DIR)
        test_ds = self.create_tf_dataset(self.config.TEST_DIR, shuffle=False)

        self.logger.info(f">> Train nb img: {tf.data.experimental.cardinality(train_ds).numpy()*self.config.BATCH_SIZE}")
        self.logger.info(f">> Val nb img: {tf.data.experimental.cardinality(val_ds).numpy()*self.config.BATCH_SIZE}")
        self.logger.info(f">> Test nb img: {tf.data.experimental.cardinality(test_ds).numpy()*self.config.BATCH_SIZE}")

        # Get class information
        num_classes = len(train_ds.class_names)
        class_names = train_ds.class_names

        # Transform datasets
        train_ds = train_ds.map(self.normalize_img).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.map(self.normalize_img).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.map(self.normalize_img).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        return {
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds
        }, num_classes, class_names

    def check_dataset_exists(self):
        """Check if dataset directories exist."""
        if os.path.exists(self.config.TRAIN_DIR) and os.path.exists(self.config.VAL_DIR) and os.path.exists(
                self.config.TEST_DIR):
            self.logger.info('Dataset exists')
            return True
        else:
            self.logger.info('Dataset does not exist')
            return False


class ModelBuilder:
    """Handles model creation and architecture."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    @staticmethod
    def data_augmentation():
        """Creates a data augmentation layer for training."""
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ])

    def build_cnn_model(self, num_classes):
        """Build the CNN model with the specified architecture."""

        # Create an empty Sequential model first
        model = keras.Sequential()

        # Add layers one by one using the add() method
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',
                                      input_shape=(self.config.IMG_SIZE[0], self.config.IMG_SIZE[1], 3)))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(num_classes, activation='softmax'))

        return model

    @staticmethod
    def create_quantized_model(model):
        """Create a quantization-aware model from an existing model."""
        return tfmot.quantization.keras.quantize_model(model)

    def compile_model(self, model, learning_rate=None):
        """Compile model with specified optimizer and metrics."""
        if learning_rate is None:
            learning_rate = self.config.LEARNING_RATE

        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


class AccuracyThresholdCallback(keras.callbacks.Callback):
    """Callback to stop training when a specific accuracy threshold is reached."""

    def __init__(self, threshold=0.95):
        super(AccuracyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_accuracy = logs.get('val_accuracy')
        if val_accuracy is not None and val_accuracy >= self.threshold:
            print(f"\nReached {self.threshold:.2f} validation accuracy, stopping training!")
            self.model.stop_training = True


class Trainer:
    """Handles model training and callbacks."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def create_callbacks(self, model_name):
        """Create callbacks for model training."""
        checkpoint_dir = os.path.join(self.config.RESULT_PATH, f"checkpoints_{model_name}")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Model checkpoint to save best weights
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.weights.h5")
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )

        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=self.config.EARLY_STOP_PATIENCE,
            min_delta=self.config.EARLY_STOP_MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        )

        # Learning rate scheduler
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )

        # Add accuracy threshold callback
        accuracy_threshold = AccuracyThresholdCallback(threshold=0.95)

        return [checkpoint_callback, early_stopping, lr_scheduler, accuracy_threshold], checkpoint_path

    def train_model(self, model, train_ds, val_ds, epochs, model_name):
        """Train model and save history."""
        callbacks, checkpoint_path = self.create_callbacks(model_name)

        self.logger.info(f"\n--- Training model ({model_name}) ---")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks
        )

        # Ensure the best weights are loaded
        if os.path.exists(checkpoint_path):
            model.load_weights(checkpoint_path)

        # Save training history plots
        self.save_model_metrics(history, model_name)

        return model, history, checkpoint_path

    def save_model_metrics(self, history, model_name):
        """Save training history plots."""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULT_PATH, f"{model_name}_history.png"))


class Evaluator:
    """Handles model evaluation and conversion."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def evaluate_model(self, model, datasets, model_name):
        """Evaluate model on train, validation, and test datasets."""
        results = {}
        for name, ds in datasets.items():
            loss, accuracy = model.evaluate(ds)
            self.logger.info(f'{name} accuracy: {accuracy:.4f}')
            results[name] = {"loss": loss, "accuracy": accuracy}

        # Save evaluation results
        results_file = os.path.join(self.config.RESULT_PATH, f"{model_name}_evaluation.txt")
        with open(results_file, 'w') as f:
            for name, metrics in results.items():
                f.write(f"{name} - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}\n")

        return results

    def convert_to_tflite(self, model, model_path, quantize=False):
        """Convert Keras model to TFLite format."""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        with open(model_path, 'wb') as f:
            f.write(tflite_model)
        self.logger.info(f"TFLite model saved to: {model_path}")

        return model_path

    def compare_model_sizes(self, models_dict):
        """Compare the size of different model formats."""
        size_comparison = {}
        for name, path in models_dict.items():
            size_bytes = os.path.getsize(path)
            size_mb = size_bytes / (1024 * 1024)
            size_comparison[name] = size_mb
            self.logger.info(f"Model: {name}, Size: {size_mb:.2f} MB")

        # Save size comparison
        with open(os.path.join(self.config.RESULT_PATH, "model_size_comparison.txt"), 'w') as f:
            for name, size in size_comparison.items():
                f.write(f"{name}: {size:.2f} MB\n")

        return size_comparison


class Utils:
    """Utility functions for the project."""

    @staticmethod
    def check_gpu():
        """Check for GPU availability."""
        print(f"TensorFlow version: {tf.__version__}")
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        print("List of GPUs Available: ", tf.config.list_physical_devices('GPU'))


def main():
    """The Main function for training and evaluating plant disease models."""
    # Initialize configuration
    config = Config('Coffee')

    # Initialize logger
    logger = Logger(config.LOG_PATH, f"tf_create_models_{config.DATASET_TYPE}")
    logger.info(f"Starting creation of TF models - Dataset {config.DATASET_TYPE}")

    # Check environment
    Utils.check_gpu()
    config.create_directories()

    # Initialize components
    data_loader = DataLoader(config, logger)
    model_builder = ModelBuilder(config, logger)
    trainer = Trainer(config, logger)
    evaluator = Evaluator(config, logger)

    # Check dataset
    if not data_loader.check_dataset_exists():
        logger.info("Dataset not found. Please run data preparation script first.")
        return

    # Load datasets
    logger.info("Creating dataset objects...")
    datasets, num_classes, class_names = data_loader.load_datasets()
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class names: {class_names}")

    # Build and train the base model
    logger.info("Creating and compiling model...")
    model_name = f"cnn_" + config.MODEL_BASE_NAME
    model = model_builder.build_cnn_model(num_classes)
    model_builder.compile_model(model)

    logger.info("Fitting model...")
    start_time = time.time()
    tracemalloc.start()
    model, history, checkpoint_path = trainer.train_model(
        model, datasets["train"], datasets["validation"], config.EPOCHS, model_name
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.info(f"Fit time: {time.time() - start_time:.2f} seconds")
    logger.info(f"Fit memory usage: {current / (1024 * 1024):.2f} MiB; Peak: {peak / (1024 * 1024):.2f} MiB")

    # Model summary
    model.summary()

    # Save the entire model
    logger.info("Saving the entire model...")
    base_model_path = os.path.join(config.MODEL_PATH, f"{model_name}.keras")
    model.save(base_model_path)

    # Evaluate the base model
    logger.info("\n--- Evaluating base model ---")
    base_results = evaluator.evaluate_model(model, datasets, model_name)

    # Convert to TFLite
    logger.info("\n--- Converting to TFLite model ---")
    tflite_model_path = os.path.join(config.MODEL_PATH, f"{model_name}.tflite")
    evaluator.convert_to_tflite(model, tflite_model_path)

    # Create and train the quantization-aware model
    logger.info("\n--- Creating quantization-aware model ---")
    quant_model_name = f"cnn_quant_" + config.MODEL_BASE_NAME
    quant_aware_model = model_builder.create_quantized_model(model)
    model_builder.compile_model(quant_aware_model, config.LEARNING_RATE)

    # Train quantization-aware model
    logger.info("Fitting quantization-aware model...")
    start_time = time.time()
    tracemalloc.start()
    quant_epochs = config.EPOCHS
    quant_aware_model, quant_history, quant_checkpoint_path = trainer.train_model(
        quant_aware_model, datasets["train"], datasets["validation"], quant_epochs, quant_model_name
    )
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.info(f"Quant fit time: {time.time() - start_time:.2f} seconds")
    logger.info(f"Quant fit memory usage: {current / (1024 * 1024):.2f} MiB; Peak: {peak / (1024 * 1024):.2f} MiB")

    # Save the quantized model
    logger.info("Saving the quantization-aware model...")
    quant_model_path = os.path.join(config.MODEL_PATH, f"{quant_model_name}.h5")
    quant_aware_model.save(quant_model_path)

    # Evaluate quantized model
    logger.info("\n--- Evaluating quantization-aware model ---")
    quant_results = evaluator.evaluate_model(quant_aware_model, datasets, quant_model_name)

    # Convert quantized model to TFLite
    logger.info("\n--- Converting to quantized TFLite model ---")
    tflite_quant_model_path = os.path.join(config.MODEL_PATH, f"{quant_model_name}.tflite")
    evaluator.convert_to_tflite(quant_aware_model, tflite_quant_model_path, quantize=True)

    # Compare model sizes
    logger.info("\n--- Comparing model sizes ---")
    models_dict = {
        "Base Model (H5)": base_model_path,
        "Base Model (TFLite)": tflite_model_path,
        "Quantized Model (H5)": quant_model_path,
        "Quantized Model (TFLite)": tflite_quant_model_path
    }
    evaluator.compare_model_sizes(models_dict)

    # Compare model accuracies for training and validation
    logger.info("\n--- Comparing model accuracies ---")
    for name, results in base_results.items():
        logger.info(f"{name} - Base Model Accuracy: {results['accuracy']:.4f}")
    for name, results in quant_results.items():
        logger.info(f"{name} - Quantized Model Accuracy: {results['accuracy']:.4f}")
    logger.info("\n--- Model accuracies compared ---")

    # Print the final summary
    logger.info("\n=== Training and Conversion Complete ===")
    logger.info(f"Models saved to: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()