"""Model management for training and evaluation."""

import os
import time
import tracemalloc
from tensorflow import keras

class ModelManager:
    """Handles model creation, training, and evaluation."""
    
    def __init__(self, config, logger, model_builder, trainer, evaluator):
        """Initialize the model manager with necessary components."""
        self.config = config
        self.logger = logger
        self.model_builder = model_builder
        self.trainer = trainer
        self.evaluator = evaluator
        self.models = {}
        self.results = {}
    
    def build_and_train_model(self, model_type, datasets, num_classes):
        """Build and train a model of the specified type."""
        model_name = f"{model_type}_" + self.config.MODEL_BASE_NAME
        model_dir = self.config.MODEL_PATH
        model_path = os.path.join(model_dir, f"{model_name}.keras")
        
        self.logger.info(f"Searching for model: {model_name}")
        self.logger.info(f"Using model: {model_dir}")
        self.logger.info(f"Model path: {model_path}")
        self.logger.info(f"Device: {self.config.DEVICE}")
        
        if os.path.exists(model_path):
            self.logger.info(f"Model {model_name} already exists. Loading model.")
            model = keras.models.load_model(model_path)
        else:
            self.logger.info(f"Creating and compiling {model_type} model...")
            
            if model_type == 'cnn':
                model = self.model_builder.build_cnn_model(num_classes)
            else:  # mobilenet
                model = self.model_builder.build_mobilenet_model(num_classes)
                
            self.model_builder.compile_model(model)

            self.logger.info(f"Fitting {model_type} model...")
            start_time = time.time()
            tracemalloc.start()
            
            model, history, checkpoint_path = self.trainer.train_model(
                model, datasets["train"], datasets["validation"], self.config.EPOCHS, model_name
            )
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.logger.info(f"Fit time: {time.time() - start_time:.2f} seconds")
            self.logger.info(f"Fit memory usage: {current / (1024 * 1024):.2f} MiB; Peak: {peak / (1024 * 1024):.2f} MiB")

            # Save the model
            self.logger.info(f"Saving the {model_type} model...")
            model.save(model_path)
        
        # Store model and path
        self.models[model_type] = {
            'model': model,
            'path': model_path,
            'name': model_name
        }
        
        return model, model_path, model_name
    
    def evaluate_model(self, model_type, datasets):
        """Evaluate a model of the specified type."""
        if model_type not in self.models:
            self.logger.error(f"Model {model_type} not found for evaluation.")
            return None
            
        model = self.models[model_type]['model']
        model_name = self.models[model_type]['name']
        
        self.logger.info(f"\n--- Evaluating {model_type} model ---")
        results = self.evaluator.evaluate_model(model, datasets, model_name)
        
        # Store results
        self.results[model_type] = results
        
        return results
    
    def create_tflite_model(self, model_type, quantize=False):
        """Create a TFLite version of the specified model."""
        if model_type not in self.models:
            self.logger.error(f"Model {model_type} not found for TFLite conversion.")
            return None
            
        model = self.models[model_type]['model']
        model_name = self.models[model_type]['name']
        
        self.logger.info(f"\n--- Converting {model_type} to TFLite model {' (quantized)' if quantize else ''} ---")
        tflite_model_path = os.path.join(self.config.MODEL_PATH, f"{model_name}.tflite")
        self.evaluator.convert_to_tflite(model, tflite_model_path, quantize=quantize)
        
        # Store TFLite path
        key = f"{model_type}_tflite{'_quant' if quantize else ''}"
        self.models[key] = {
            'path': tflite_model_path,
            'name': model_name
        }
        
        return tflite_model_path
    
    def create_quantized_model(self, model_type, datasets):
        """Create and train a quantized version of the specified model."""
        if model_type not in self.models:
            self.logger.error(f"Base model {model_type} not found for quantization.")
            return None
            
        base_model = self.models[model_type]['model']
        quant_model_name = f"{model_type}_quant_" + self.config.MODEL_BASE_NAME
        quant_model_path = os.path.join(self.config.MODEL_PATH, f"{quant_model_name}.h5")
        
        self.logger.info(f"\n--- Creating quantization-aware {model_type} model ---")
        
        if os.path.exists(quant_model_path):
            self.logger.info(f"Quantized model {quant_model_name} already exists. Loading model.")
            quant_model = keras.models.load_model(quant_model_path)
        else:
            quant_model = self.model_builder.create_quantized_model(base_model)
            self.model_builder.compile_model(quant_model, self.config.LEARNING_RATE)

            # Train quantization-aware model
            self.logger.info(f"Fitting quantization-aware {model_type} model...")
            start_time = time.time()
            tracemalloc.start()
            
            quant_model, quant_history, quant_checkpoint_path = self.trainer.train_model(
                quant_model, datasets["train"], datasets["validation"], self.config.EPOCHS, quant_model_name
            )
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.logger.info(f"Quant fit time: {time.time() - start_time:.2f} seconds")
            self.logger.info(f"Quant fit memory usage: {current / (1024 * 1024):.2f} MiB; Peak: {peak / (1024 * 1024):.2f} MiB")

            # Save the quantized model
            self.logger.info(f"Saving the quantization-aware {model_type} model...")
            quant_model.save(quant_model_path)
        
        # Store quantized model and path
        quant_key = f"{model_type}_quant"
        self.models[quant_key] = {
            'model': quant_model,
            'path': quant_model_path,
            'name': quant_model_name
        }
        
        return quant_model, quant_model_path, quant_model_name
    
    def compare_model_sizes(self):
        """Compare the sizes of all created models."""
        if not self.models:
            self.logger.warning("No models available for size comparison.")
            return
            
        self.logger.info("\n--- Comparing model sizes ---")
        
        models_dict = {}
        for key, model_info in self.models.items():
            if 'path' in model_info:
                display_name = key.replace('_', ' ').title()
                if 'tflite' in key:
                    display_name += ' (TFLite)'
                models_dict[display_name] = model_info['path']
        
        self.evaluator.compare_model_sizes(models_dict)
    
    def compare_model_accuracies(self):
        """Compare the accuracies of all evaluated models."""
        if not self.results:
            self.logger.warning("No evaluation results available for comparison.")
            return
            
        self.logger.info("\n--- Comparing model accuracies ---")
        
        for model_type, results in self.results.items():
            display_name = model_type.replace('_', ' ').title()
            for dataset_name, metrics in results.items():
                self.logger.info(f"{dataset_name} - {display_name} Model Accuracy: {metrics['accuracy']:.4f}")