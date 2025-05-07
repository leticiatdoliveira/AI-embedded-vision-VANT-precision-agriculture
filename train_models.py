"""Main script for training and evaluating plant disease models."""

# TODO: add ep as arg

import os
import argparse
from tensorflow import keras
from logger import Logger
from config import Config
from data_loader import DataLoader
from model_builder import ModelBuilder
from trainer import Trainer
from evaluator import Evaluator
from utils import Utils
from menu_system import MenuSystem
from model_manager import ModelManager

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all messages, 1 = INFO, 2 = WARNING, 3 = ERROR


def main():
    """The Main function for training and evaluating plant disease models."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate plant disease models.')
    parser.add_argument('--dataset', type=str, default='Coffee', 
                        help='Dataset name (e.g., Coffee, PlantVillage)')
    parser.add_argument('--device', type=str, default='rasp',
                        help='Device type (e.g., rasp, desktop)')
    args = parser.parse_args()
    
    # Initialize configuration with command line arguments
    config = Config(args.dataset)
    config.DEVICE = args.device  # Set device type in config
    device = args.device

    # Initialize logger
    logger = Logger(config.LOG_PATH, f"tf_create_models_{device}_{config.DATASET_TYPE}_{config.MODEL_BASE_NAME}")
    logger.info(f"Starting creation of TF models - Dataset {config.DATASET_TYPE}")
    logger.info(f"Using device: {device}")

    # Check environment
    Utils.check_gpu()
    config.create_directories()

    # Initialize components
    data_loader = DataLoader(config, logger)
    model_builder = ModelBuilder(config, logger)
    trainer = Trainer(config, logger)
    evaluator = Evaluator(config, logger)
    
    # Initialize menu system
    menu = MenuSystem(logger)
    
    # Initialize model manager
    model_manager = ModelManager(config, logger, model_builder, trainer, evaluator)

    # Check dataset
    if not data_loader.check_dataset_exists():
        logger.info("Dataset not found. Please run data preparation script first.")
        return

    # Load datasets
    logger.info("Creating dataset objects...")
    datasets, num_classes, class_names = data_loader.load_datasets()
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class names: {class_names}")
    
    # Load MobileNet datasets if needed
    mobilenet_datasets = None
    
    # Get user choices from menu
    choices = menu.get_all_choices()
    model_type = choices['model_type']
    create_tflite = choices['create_tflite']
    create_quant = choices['create_quant']
    create_quant_tflite = choices['create_quant_tflite']
    
    # Prepare appropriate datasets based on model type
    if model_type == 'mobilenet':
        logger.info("Creating MobileNet-specific dataset objects...")
        mobilenet_datasets, _, _ = data_loader.load_mobilenet_datasets()
        current_datasets = mobilenet_datasets
    else:
        current_datasets = datasets
    
    # Build and train the selected model
    model, model_path, model_name = model_manager.build_and_train_model(
        model_type, current_datasets, num_classes
    )
    
    # Model summary
    model.summary()
    
    # Evaluate the model
    model_manager.evaluate_model(model_type, current_datasets)
    
    # Create TFLite model if requested
    if create_tflite:
        model_manager.create_tflite_model(model_type)
    
    # Create quantized model if requested
    if create_quant:
        quant_model, quant_model_path, quant_model_name = model_manager.create_quantized_model(
            model_type, current_datasets
        )
        
        # Evaluate quantized model
        quant_key = f"{model_type}_quant"
        model_manager.evaluate_model(quant_key, current_datasets)
        
        # Create quantized TFLite model if requested
        if create_quant_tflite:
            model_manager.create_tflite_model(quant_key, quantize=True)
    
    # Compare model sizes
    model_manager.compare_model_sizes()
    
    # Compare model accuracies
    model_manager.compare_model_accuracies()
    
    # Print the final summary
    logger.info("\n=== Training and Conversion Complete ===")
    logger.info(f"Models saved to: {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
