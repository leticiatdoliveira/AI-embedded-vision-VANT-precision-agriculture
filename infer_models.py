import os
import logging
import numpy as np
import tensorflow as tf
import time
import tracemalloc
from PIL import Image
import matplotlib.pyplot as plt
from logger import Logger
import tensorflow_model_optimization as tfmot

class Config:
    """Configuration parameters for model inference."""

    def __init__(self, dataset_type, set_type, epochs):
        # Constants
        self.IMG_SIZE = (224, 224)
        self.BATCH_SIZE = 32
        self.EPOCHS = epochs
        self.LEARNING_RATE = 0.001
        self.DATA_AUGMENTATION = False
        self.DATASET_TYPE = dataset_type
        self.SET_TYPE = set_type
        self.MODEL_BASE_NAME = f"ep={self.EPOCHS}_lr={self.LEARNING_RATE}_aug={self.DATA_AUGMENTATION}"
        self.MODEL_NAME = f"cnn_{self.MODEL_BASE_NAME}.keras"
        self.TFLITE_MODEL_NAME = f"cnn_{self.MODEL_BASE_NAME}.tflite"
        self.QUANT_MODEL_NAME = f"cnn_quant_{self.MODEL_BASE_NAME}.h5"
        self.QUANT_LITE_MODEL_NAME = f"cnn_quant_{self.MODEL_BASE_NAME}.tflite"
        self.SEED = 42

        # Path setup
        self.ROOT_PATH = os.getcwd()
        self.PROJECT_PATH = os.path.join(self.ROOT_PATH, self.DATASET_TYPE)
        self.DATASET_PATH = os.path.join(self.PROJECT_PATH, 'dataset')
        self.MODEL_PATH = os.path.join(self.PROJECT_PATH, 'model')
        self.RESULT_PATH = os.path.join(self.PROJECT_PATH, 'results')
        self.INFERENCES_PATH = os.path.join(self.RESULT_PATH, 'inferences')
        self.LOG_PATH = os.path.join(self.PROJECT_PATH, 'log')

        # Dataset paths
        self.TEST_DIR = os.path.join(self.DATASET_PATH, self.SET_TYPE)

        # Timestamp for organization
        self.TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
        self.INFERENCE_SESSION = f"inference_{self.TIMESTAMP}"
        self.CURRENT_INFERENCE_PATH = os.path.join(self.INFERENCES_PATH, self.INFERENCE_SESSION)

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        for path in [self.PROJECT_PATH, self.DATASET_PATH, self.MODEL_PATH,
                     self.RESULT_PATH, self.INFERENCES_PATH, self.CURRENT_INFERENCE_PATH,
                     self.LOG_PATH]:
            if not os.path.exists(path):
                print(f"Creating directory {path}")
                os.makedirs(path)


class DataLoader:
    """Handles dataset loading and preprocessing for inference."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def normalize_img(self, image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    def create_tf_dataset(self, dataset_path, label_type='inferred', label_mode='categorical', shuffle=False):
        """Create a TensorFlow dataset from directory."""
        try:
            dataset = tf.keras.utils.image_dataset_from_directory(
                dataset_path,
                labels=label_type,
                label_mode=label_mode,
                image_size=self.config.IMG_SIZE,
                batch_size=self.config.BATCH_SIZE,
                shuffle=shuffle,
                seed=self.config.SEED
            )
            return dataset
        except Exception as e:
            self.logger.error(f"Error creating dataset from {dataset_path}: {str(e)}")
            return None

    def load_test_dataset(self):
        """Load and prepare test dataset."""
        try:
            self.logger.info(f"Loading test dataset from {self.config.TEST_DIR}")
            test_ds = self.create_tf_dataset(self.config.TEST_DIR)

            if test_ds is None:
                self.logger.error("Failed to load test dataset")
                return None, None, None

            # Get class information
            num_classes = len(test_ds.class_names)
            class_names = test_ds.class_names

            # Transform dataset
            test_ds = test_ds.map(self.normalize_img).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

            self.logger.info(
                f"Test dataset loaded with {tf.data.experimental.cardinality(test_ds).numpy() * self.config.BATCH_SIZE} images")
            self.logger.info(f"Number of classes: {num_classes}")
            self.logger.info(f"Class names: {class_names}")

            return test_ds, num_classes, class_names
        except Exception as e:
            self.logger.error(f"Error in load_test_dataset: {str(e)}")
            return None, None, None


class ModelInference:
    """Handles loading and inference for both TFLite and Keras models."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_model(self, model_path):
        """Load model based on file extension."""
        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            return None

        try:
            if model_path.endswith('.tflite'):
                return self.load_tflite_model(model_path)
            elif model_path.endswith(('.h5', '.keras')):
                return self.load_keras_model(model_path)
            else:
                self.logger.error(f"Unsupported model format: {model_path}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading model {model_path}: {str(e)}")
            return None

    def load_tflite_model(self, model_path):
        """Load TFLite model from file."""
        try:
            self.logger.info(f"Loading TFLite model from {model_path}")
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return {'type': 'tflite', 'model': interpreter}
        except Exception as e:
            self.logger.error(f"Error loading TFLite model: {str(e)}")
            return None

    def load_keras_model(self, model_path):
        """Load Keras model from file."""
        try:
            self.logger.info(f"Loading Keras model from {model_path}")
            import tensorflow as tf
            import tensorflow_model_optimization as tfmot

            # For regular model (.keras file)
            if model_path.endswith('.keras'):
                try:
                    # Recreate the model architecture and load weights directly
                    model = self.recreate_model()
                    model.load_weights(model_path)
                    return {'type': 'keras', 'model': model}
                except Exception as e:
                    self.logger.error(f"Error loading .keras model weights: {str(e)}")
                    return None

            # For quantized model (.h5 file)
            elif model_path.endswith('.h5'):
                try:
                    # Register the quantization layers with custom_objects
                    with tf.keras.utils.custom_object_scope({
                        'QuantizeLayer': tfmot.quantization.keras.QuantizeLayer,
                        'QuantizeWrapper': tfmot.quantization.keras.QuantizeWrapper,
                        'NoOpQuantizeLayer': tfmot.quantization.keras.NoOpQuantizeLayer
                    }):
                        model = tf.keras.models.load_model(model_path, compile=False)
                    return {'type': 'keras', 'model': model}
                except Exception as e:
                    self.logger.error(f"Error loading quantized model: {str(e)}")
                    # Fall back to recreating the model
                    try:
                        base_model = self.recreate_model()
                        quant_model = tfmot.quantization.keras.quantize_model(base_model)
                        quant_model.load_weights(model_path)
                        return {'type': 'keras', 'model': quant_model}
                    except Exception as e2:
                        self.logger.error(f"Error recreating quantized model: {str(e2)}")
                        return None
        except Exception as e:
            self.logger.error(f"Error loading Keras model: {str(e)}")
            return None

    @staticmethod
    def recreate_model():
        """Recreate the model architecture."""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        return model

    def run_inference(self, model_info, test_ds, class_names):
        """Run inference based on model type."""
        if model_info is None:
            self.logger.error("Cannot run inference: No model provided")
            return None

        if model_info['type'] == 'tflite':
            return self.run_tflite_inference(model_info['model'], test_ds, class_names)
        elif model_info['type'] == 'keras':
            return self.run_keras_inference(model_info['model'], test_ds, class_names)
        else:
            self.logger.error(f"Unsupported model type: {model_info['type']}")
            return None

    def run_tflite_inference(self, interpreter, test_ds, class_names):
        """Run inference using TFLite model on test dataset."""
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        all_predictions = []
        all_labels = []

        start_time = time.time()
        tracemalloc.start()

        try:
            # Iterate through the test dataset
            for images, labels in test_ds:
                for i in range(len(images)):
                    # Pre-process the image
                    img = images[i].numpy()
                    img = np.expand_dims(img, axis=0)

                    # Set the tensor
                    interpreter.set_tensor(input_index, img)

                    # Run inference
                    interpreter.invoke()

                    # Get the output
                    output = interpreter.get_tensor(output_index)

                    # Process prediction
                    pred = np.argmax(output[0])
                    all_predictions.append(pred)

                    # Process label
                    label = np.argmax(labels[i].numpy())
                    all_labels.append(label)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            inference_time = time.time() - start_time

            # Calculate accuracy
            accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_predictions)

            self.logger.info(f"TFLite inference completed in {inference_time:.2f} seconds")
            self.logger.info(f"Memory usage: {current / (1024 * 1024):.2f} MB; Peak: {peak / (1024 * 1024):.2f} MB")
            self.logger.info(f"Accuracy: {accuracy:.4f}")

            return {
                "accuracy": accuracy,
                "inference_time": inference_time,
                "memory_current": current / (1024 * 1024),
                "memory_peak": peak / (1024 * 1024),
                "predictions": all_predictions,
                "labels": all_labels
            }
        except Exception as e:
            self.logger.error(f"Error during TFLite inference: {str(e)}")
            return None

    def run_keras_inference(self, model, test_ds, class_names):
        """Run inference using Keras model on test dataset."""
        all_predictions = []
        all_labels = []

        start_time = time.time()
        tracemalloc.start()

        try:
            # Get predictions for the entire dataset at once
            predictions = model.predict(test_ds)

            # Process predictions and labels
            for images, labels in test_ds:
                for i in range(len(images)):
                    if len(all_predictions) < len(predictions):
                        # Process prediction
                        pred_idx = len(all_predictions)
                        pred = np.argmax(predictions[pred_idx])
                        all_predictions.append(pred)

                        # Process label
                        label = np.argmax(labels[i].numpy())
                        all_labels.append(label)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            inference_time = time.time() - start_time

            # Calculate accuracy
            accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_predictions)

            self.logger.info(f"Keras inference completed in {inference_time:.2f} seconds")
            self.logger.info(f"Memory usage: {current / (1024 * 1024):.2f} MB; Peak: {peak / (1024 * 1024):.2f} MB")
            self.logger.info(f"Accuracy: {accuracy:.4f}")

            return {
                "accuracy": accuracy,
                "inference_time": inference_time,
                "memory_current": current / (1024 * 1024),
                "memory_peak": peak / (1024 * 1024),
                "predictions": all_predictions,
                "labels": all_labels
            }
        except Exception as e:
            self.logger.error(f"Error during Keras inference: {str(e)}")
            return None

    def visualize_results(self, results, model_name):
        """Create and save visualization of inference results."""
        if results is None:
            self.logger.error("Cannot visualize results: No results provided")
            return

        try:
            # Create accuracy bar chart
            plt.figure(figsize=(10, 6))
            plt.bar(['Accuracy'], [results["accuracy"]], color='blue')
            plt.title(f'Model Accuracy: {model_name}')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)

            # Save the figure
            result_path = os.path.join(self.config.CURRENT_INFERENCE_PATH, f"{model_name}_accuracy.png")
            plt.savefig(result_path)
            plt.close()
            self.logger.info(f"Results visualization saved to {result_path}")

            # Save metrics to text file
            metrics_path = os.path.join(self.config.CURRENT_INFERENCE_PATH, f"{model_name}_metrics.txt")
            with open(metrics_path, 'w') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Accuracy: {results['accuracy']:.4f}\n")
                f.write(f"Inference Time: {results['inference_time']:.2f} seconds\n")
                f.write(f"Memory Usage (Current): {results['memory_current']:.2f} MB\n")
                f.write(f"Memory Usage (Peak): {results['memory_peak']:.2f} MB\n")

            self.logger.info(f"Metrics saved to {metrics_path}")
        except Exception as e:
            self.logger.error(f"Error visualizing results: {str(e)}")


def compare_model_results(results_dict, config, logger):
    """Compare and visualize results from different models."""
    if len(results_dict) <= 1:
        logger.info("Not enough models to compare")
        return

    try:
        # Extract performance metrics
        accuracies = {name: result["accuracy"] for name, result in results_dict.items()}
        inf_times = {name: result["inference_time"] for name, result in results_dict.items()}
        memory_usage = {name: result["memory_peak"] for name, result in results_dict.items()}

        # Create comparative bar charts
        # 1. Accuracy comparison
        plt.figure(figsize=(12, 6))
        plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
        plt.title('Accuracy Comparison Between Models')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        for i, v in enumerate(accuracies.values()):
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
        plt.tight_layout()
        accuracy_path = os.path.join(config.CURRENT_INFERENCE_PATH, "model_accuracy_comparison.png")
        plt.savefig(accuracy_path)
        plt.close()

        # 2. Inference time comparison
        plt.figure(figsize=(12, 6))
        plt.bar(inf_times.keys(), inf_times.values(), color='lightgreen')
        plt.title('Inference Time Comparison Between Models')
        plt.ylabel('Time (seconds)')
        for i, v in enumerate(inf_times.values()):
            plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')
        plt.tight_layout()
        time_path = os.path.join(config.CURRENT_INFERENCE_PATH, "model_inference_time_comparison.png")
        plt.savefig(time_path)
        plt.close()

        # 3. Memory usage comparison
        plt.figure(figsize=(12, 6))
        plt.bar(memory_usage.keys(), memory_usage.values(), color='salmon')
        plt.title('Memory Usage Comparison Between Models')
        plt.ylabel('Memory (MB)')
        for i, v in enumerate(memory_usage.values()):
            plt.text(i, v + 0.5, f"{v:.2f}MB", ha='center')
        plt.tight_layout()
        memory_path = os.path.join(config.CURRENT_INFERENCE_PATH, "model_memory_comparison.png")
        plt.savefig(memory_path)
        plt.close()

        # Create summary visualization
        plt.figure(figsize=(15, 10))

        # Normalize metrics for fair comparison in a single chart
        norm_times = {k: v / max(inf_times.values()) for k, v in inf_times.items()}
        norm_memory = {k: v / max(memory_usage.values()) for k, v in memory_usage.items()}

        x = np.arange(len(accuracies))
        width = 0.25

        plt.bar(x - width, accuracies.values(), width, label='Accuracy', color='skyblue')
        plt.bar(x, norm_times.values(), width, label='Normalized Time', color='lightgreen')
        plt.bar(x + width, norm_memory.values(), width, label='Normalized Memory', color='salmon')

        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, accuracies.keys())
        plt.legend()
        plt.tight_layout()

        summary_path = os.path.join(config.CURRENT_INFERENCE_PATH, "model_overall_comparison.png")
        plt.savefig(summary_path)
        plt.close()

        # Save comparison to text file
        comparison_path = os.path.join(config.CURRENT_INFERENCE_PATH, "model_comparison.txt")
        with open(comparison_path, 'w') as f:
            f.write("MODEL COMPARISON RESULTS\n")
            f.write("=======================\n\n")

            f.write("ACCURACY\n")
            for name, acc in accuracies.items():
                f.write(f"{name}: {acc:.4f}\n")
            f.write("\n")

            f.write("INFERENCE TIME\n")
            for name, time_val in inf_times.items():
                f.write(f"{name}: {time_val:.2f} seconds\n")
            f.write("\n")

            f.write("MEMORY USAGE (PEAK)\n")
            for name, mem in memory_usage.items():
                f.write(f"{name}: {mem:.2f} MB\n")

            # Create a summary file
            summary_file = os.path.join(config.CURRENT_INFERENCE_PATH, "summary.txt")
            with open(summary_file, 'w') as summary:
                summary.write(f"Inference Session: {config.INFERENCE_SESSION}\n")
                summary.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                summary.write(f"Dataset: {config.DATASET_TYPE}\n")
                summary.write(f"Set type: {config.SET_TYPE}\n")
                summary.write(f"Number of models compared: {len(results_dict)}\n\n")

                # Best model identification
                best_acc_model = max(accuracies.items(), key=lambda x: x[1])[0]
                fastest_model = min(inf_times.items(), key=lambda x: x[1])[0]
                lowest_memory = min(memory_usage.items(), key=lambda x: x[1])[0]

                summary.write(f"Most accurate model: {best_acc_model} ({accuracies[best_acc_model]:.4f})\n")
                summary.write(f"Fastest model: {fastest_model} ({inf_times[fastest_model]:.2f}s)\n")
                summary.write(f"Lowest memory usage: {lowest_memory} ({memory_usage[lowest_memory]:.2f}MB)\n")

        logger.info(f"Model comparison results saved to {comparison_path}")
        logger.info(f"Model comparison visualizations saved to {config.CURRENT_INFERENCE_PATH}")

    except Exception as e:
        logger.error(f"Error comparing model results: {str(e)}")


def main():
    """Main function to run TF models inference."""
    # Initialize configuration
    config = Config('Coffee', 'test', 20)
    config.create_directories()

    # Initialize logger
    logger = Logger(config.LOG_PATH, f"tf_models_inference_{config.SET_TYPE}")
    logger.info(f"Starting TF models inference - Dataset {config.DATASET_TYPE} - Set {config.SET_TYPE}")

    # Initialize DataLoader
    data_loader = DataLoader(config, logger)

    # Load test dataset
    test_ds, num_classes, class_names = data_loader.load_test_dataset()
    if test_ds is None:
        logger.error("Failed to load test dataset. Exiting.")
        return
    logger.info(f"Test dataset loaded ! - {num_classes} classes - {len(test_ds) * config.BATCH_SIZE} images")

    # Initialize Inference
    model_inference = ModelInference(config, logger)

    # Dictionary to store results for comparison
    all_results = {}

    # Load and evaluate Keras model
    keras_model_path = os.path.join(config.MODEL_PATH, config.MODEL_NAME)
    if os.path.exists(keras_model_path):
        logger.info(f"Processing Keras model: {config.MODEL_NAME}")
        keras_model_info = model_inference.load_model(keras_model_path)
        if keras_model_info:
            keras_results = model_inference.run_inference(keras_model_info, test_ds, class_names)
            model_inference.visualize_results(keras_results, "keras_model")
            all_results["keras"] = keras_results
    else:
        logger.warning(f"Keras model not found at {keras_model_path}")

    # Load and evaluate regular TFLite model
    tflite_model_path = os.path.join(config.MODEL_PATH, config.TFLITE_MODEL_NAME)
    if os.path.exists(tflite_model_path):
        logger.info(f"Processing TFLite model: {config.TFLITE_MODEL_NAME}")
        tflite_model_info = model_inference.load_model(tflite_model_path)
        if tflite_model_info:
            tflite_results = model_inference.run_inference(tflite_model_info, test_ds, class_names)
            model_inference.visualize_results(tflite_results, "tflite_model")
            all_results["tflite"] = tflite_results
    else:
        logger.warning(f"TFLite model not found at {tflite_model_path}")

    # Load and evaluate quantized models
    # First the H5/keras quantized model
    quant_model_path = os.path.join(config.MODEL_PATH, config.QUANT_MODEL_NAME)
    if os.path.exists(quant_model_path):
        logger.info(f"Processing quantized Keras model: {config.QUANT_MODEL_NAME}")
        quant_model_info = model_inference.load_model(quant_model_path)
        if quant_model_info:
            quant_results = model_inference.run_inference(quant_model_info, test_ds, class_names)
            model_inference.visualize_results(quant_results, "quant_keras_model")
            all_results["quant_keras"] = quant_results
    else:
        logger.warning(f"Quantized Keras model not found at {quant_model_path}")

    # Then the quantized TFLite model
    quant_tflite_model_path = os.path.join(config.MODEL_PATH, config.QUANT_LITE_MODEL_NAME)
    if os.path.exists(quant_tflite_model_path):
        logger.info(f"Processing quantized TFLite model: {config.QUANT_LITE_MODEL_NAME}")
        quant_tflite_model_info = model_inference.load_model(quant_tflite_model_path)
        if quant_tflite_model_info:
            quant_tflite_results = model_inference.run_inference(quant_tflite_model_info, test_ds, class_names)
            model_inference.visualize_results(quant_tflite_results, "quant_tflite_model")
            all_results["quant_tflite"] = quant_tflite_results
    else:
        logger.warning(f"Quantized TFLite model not found at {quant_tflite_model_path}")

    # Compare results between models
    if all_results:
        compare_model_results(all_results, config, logger)

    logger.info("Model inference completed")


if __name__ == "__main__":
    main()
