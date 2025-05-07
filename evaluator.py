import os
import numpy as np
import tensorflow as tf

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
