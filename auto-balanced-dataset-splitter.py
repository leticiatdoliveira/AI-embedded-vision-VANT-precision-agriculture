# -*- coding: utf-8 -*-
"""
Auto-Balanced Dataset Splitter
Automatically detects smallest class size and under samples all classes to match
"""

import os
import random
import shutil
from logger import Logger


class Config:
    """Configuration class for dataset management."""

    def __init__(self, dataset_type='PlantVillage', plant_culture='Tomato', balance_classes=False):
        """
        Initialize configuration with dataset parameters.

        Args:
            dataset_type: Type of dataset ('Coffee' or 'PlantVillage')
            plant_culture: Plant culture name (only used for PlantVillage)
            balance_classes: Whether to balance classes by under-sampling
        """
        # Set seed for reproducibility
        self.SEED_VALUE = 42
        random.seed(self.SEED_VALUE)
        os.environ['PYTHONHASHSEED'] = str(self.SEED_VALUE)

        # Dataset configuration
        self.DATASET_TYPE = dataset_type
        self.PLANT_CULTURE = plant_culture
        self.BALANCE_CLASSES = balance_classes

        # Path setup
        self.root_path = os.getcwd()
        self.project_path = os.path.join(self.root_path, f'{self.DATASET_TYPE}')
        self.DATA_PATH = os.path.join(self.project_path, 'data')
        self.DATASET_PATH = os.path.join(self.project_path, 'dataset')
        self.MODEL_PATH = os.path.join(self.project_path, 'model')
        self.LOG_PATH = os.path.join(self.project_path, 'log')

        # Dataset split path
        self.TRAIN_PATH = os.path.join(self.DATASET_PATH, 'train')
        self.VALIDATION_PATH = os.path.join(self.DATASET_PATH, 'validation')
        self.TEST_PATH = os.path.join(self.DATASET_PATH, 'test')

    def create_directories(self):
        """Create all necessary directories if they don't exist."""
        for dir_path in [self.project_path, self.DATASET_PATH, self.MODEL_PATH, self.LOG_PATH,
                         self.TRAIN_PATH, self.VALIDATION_PATH, self.TEST_PATH]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Created directory: {dir_path}")


def analyze_class_distribution(source_data_path, filter_prefix=None):
    """
    Analyzes class distribution in the source data path.

    Args:
        source_data_path: Path to the source data directory
        filter_prefix: Optional prefix to filter class names (e.g., 'Tomato')

    Returns:
        Dictionary with class names as keys and image counts as values
    """
    class_counts = {}

    for class_name in os.listdir(source_data_path):
        # Skip if filter_prefix is specified and class name doesn't contain it
        if filter_prefix and filter_prefix not in class_name:
            continue

        class_dir = os.path.join(source_data_path, class_name)
        if os.path.isdir(class_dir):
            image_files = [f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                           and os.path.isfile(os.path.join(class_dir, f))]
            class_counts[class_name] = len(image_files)

    return class_counts


def find_min_class_size(class_counts):
    """
    Finds the size of the smallest class in the dataset.
    
    Args:
        class_counts: Dictionary with class names as keys and image counts as values
        
    Returns:
        Integer representing the smallest class size
    """
    if not class_counts:
        return 0, None

    min_class_size = min(class_counts.values())
    min_class_name = min(class_counts, key=class_counts.get)

    return min_class_size, min_class_name


def split_data_from_drive_balanced(config, logger, train_ratio=0.6, validation_ratio=0.2):
    """
    Splits data directly from Drive with automatic under-sampling to balance classes.
    
    Args:
        config: Configuration object
        logger: Logger object for logging messages
        train_ratio: The proportion of data to use for training.
        validation_ratio: The proportion of data to use for validation.
    """
    for dir_path in [config.TRAIN_PATH, config.VALIDATION_PATH, config.TEST_PATH]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info(f"Starting to split data from {config.DATA_PATH} with balancing={config.BALANCE_CLASSES}")

    # Check if the source directory exists
    if not os.path.exists(config.DATA_PATH):
        logger.error(f"Error: Source path {config.DATA_PATH} does not exist!")
        return

    # Analyze class distribution
    if config.DATASET_TYPE == 'PlantVillage':
        class_counts = analyze_class_distribution(config.DATA_PATH, filter_prefix=config.PLANT_CULTURE)
    else:
        class_counts = analyze_class_distribution(config.DATA_PATH)

    # Calculate statistics about class distribution
    total_images = sum(class_counts.values())
    avg_class_size = total_images / len(class_counts) if class_counts else 0
    min_class_size, min_class_name = find_min_class_size(class_counts)
    max_class_size = max(class_counts.values()) if class_counts else 0
    max_class_name = max(class_counts, key=class_counts.get) if class_counts else None

    logger.info(f"Original class distribution:")
    for class_name, count in sorted(class_counts.items()):
        logger.info(f"  {class_name}: {count} images")

    logger.info(f"Class distribution statistics:")
    logger.info(f"  Total classes: {len(class_counts)}")
    logger.info(f"  Total images: {total_images}")
    logger.info(f"  Average images per class: {avg_class_size:.2f}")
    logger.info(f"  Minimum class size: {min_class_size} ('{min_class_name}')")
    logger.info(f"  Maximum class size: {max_class_size} ('{max_class_name}')")
    logger.info(f"  Imbalance ratio (max/min): {max_class_size / min_class_size:.2f}x")

    # Set samples_per_class to the smallest class size for automatic balancing
    if config.BALANCE_CLASSES:
        samples_per_class = min_class_size
        logger.info(f"Automatically under-sampling all classes to match smallest class: {samples_per_class} images per class")
    else:
        # No balancing - use full dataset
        samples_per_class = None
        logger.info("No under-sampling applied - using full dataset")

    # Process each class directory
    for class_name in os.listdir(config.DATA_PATH):
        # Skip if filter_prefix is specified and class name doesn't contain it
        if config.PLANT_CULTURE and config.PLANT_CULTURE not in class_name:
            continue

        class_dir = os.path.join(config.DATA_PATH, class_name)
        if not os.path.isdir(class_dir):
            logger.warning(f"Warning: {class_dir} is not a directory, skipping...")
            continue

        logger.info(f"Processing class: {class_name}")

        # Get all image files in the class directory
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                       and os.path.isfile(os.path.join(class_dir, f))]

        if not image_files:
            logger.warning(f"Warning: No images found in {class_dir}")
            continue

        # Undersample if needed
        original_count = len(image_files)
        if config.BALANCE_CLASSES and samples_per_class is not None and original_count > samples_per_class:
            logger.info(f"Under-sampling class {class_name} from {original_count} to {samples_per_class} images")
            random.shuffle(image_files)
            image_files = image_files[:samples_per_class]

        # Shuffle the files for random splitting
        random.shuffle(image_files)

        # Calculate split sizes
        num_images = len(image_files)
        num_train = int(num_images * train_ratio)
        num_validation = int(num_images * validation_ratio)

        # Split the list of files
        train_images = image_files[:num_train]
        validation_images = image_files[num_train:num_train + num_validation]
        test_images = image_files[num_train + num_validation:]

        # Create destination class directories
        dst_train_class_dir = os.path.join(config.TRAIN_PATH, class_name)
        dst_validation_class_dir = os.path.join(config.VALIDATION_PATH, class_name)
        dst_test_class_dir = os.path.join(config.TEST_PATH, class_name)

        for dir_path in [dst_train_class_dir, dst_validation_class_dir, dst_test_class_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        # Copy files to their respective directories
        for image_set, dst_dir in [(train_images, dst_train_class_dir),
                                   (validation_images, dst_validation_class_dir),
                                   (test_images, dst_test_class_dir)]:
            for image_file in image_set:
                src_path = os.path.join(class_dir, image_file)
                dst_path = os.path.join(dst_dir, image_file)
                shutil.copy2(src_path, dst_path)  # copy2 preserves metadata

        logger.info(f"Class {class_name}: {len(train_images)} train, {len(validation_images)} validation, {len(test_images)} test")


def verify_dataset_stats(dataset_path):
    """
    Verifies the number of data and classes in a dataset.
    
    Args:
        dataset_path: The path to the dataset directory.
    """
    num_data = 0
    num_classes = 0
    class_counts = {}

    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            num_classes += 1
            num_files = len([f for f in os.listdir(class_dir)
                             if os.path.isfile(os.path.join(class_dir, f)) and
                             f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            num_data += num_files
            class_counts[class_name] = num_files

    return num_data, num_classes, class_counts


def check_balance_quality(class_counts):
    """
    Checks how well-balanced the dataset is after splitting.
    
    Args:
        class_counts: Dictionary with class names as keys and image counts as values
        
    Returns:
        Float representing coefficient of variation (lower is better balanced)
    """
    if not class_counts:
        return 0, 0, 0

    values = list(class_counts.values())
    mean = sum(values) / len(values)

    if mean == 0:
        return 0, 0, 0

    # Calculate standard deviation
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std_dev = variance ** 0.5

    # Coefficient of variation (measure of relative variability)
    cv = std_dev / mean

    return cv, mean, std_dev


def main():
    print("Starting auto-balanced dataset split")

    # Initialize configuration
    config = Config(
        dataset_type='PlantVillage',
        plant_culture='Tomato',
        balance_classes=False
    )
    config.create_directories()

    # Init Logger
    logger = Logger(config.LOG_PATH, f"{config.PLANT_CULTURE}_dataset_split")
    logger.info("Starting auto-balanced dataset split")

    # Split the data with automatic balancing
    split_data_from_drive_balanced(
        config=config,
        logger=logger,
        train_ratio=0.6,
        validation_ratio=0.2
    )

    # Verify split results using Config paths
    train_num_data, train_num_classes, train_class_counts = verify_dataset_stats(config.TRAIN_PATH)
    validation_num_data, validation_num_classes, validation_class_counts = verify_dataset_stats(config.VALIDATION_PATH)
    test_num_data, test_num_classes, test_class_counts = verify_dataset_stats(config.TEST_PATH)

    # Check balance quality
    train_cv, train_mean, train_std = check_balance_quality(train_class_counts)
    validation_cv, validation_mean, validation_std = check_balance_quality(validation_class_counts)
    test_cv, test_mean, test_std = check_balance_quality(test_class_counts)

    logger.info("Balance quality measures (lower CV = better balance):")
    logger.info(f"  Train: CV={train_cv:.4f}, Mean={train_mean:.2f}, StdDev={train_std:.2f}")
    logger.info(f"  Validation: CV={validation_cv:.4f}, Mean={validation_mean:.2f}, StdDev={validation_std:.2f}")
    logger.info(f"  Test: CV={test_cv:.4f}, Mean={test_mean:.2f}, StdDev={test_std:.2f}")

    # Print detailed class distribution for each split
    logger.info("Train dataset class distribution:")
    for class_name, count in sorted(train_class_counts.items()):
        logger.info(f"  {class_name}: {count} images")

    logger.info("Validation dataset class distribution:")
    for class_name, count in sorted(validation_class_counts.items()):
        logger.info(f"  {class_name}: {count} images")

    logger.info("Test dataset class distribution:")
    for class_name, count in sorted(test_class_counts.items()):
        logger.info(f"  {class_name}: {count} images")

    logger.info(f"Train Dataset: {train_num_data} images, {train_num_classes} classes")
    logger.info(f"Validation Dataset: {validation_num_data} images, {validation_num_classes} classes")
    logger.info(f"Test Dataset: {test_num_data} images, {test_num_classes} classes")

    total_data = train_num_data + validation_num_data + test_num_data
    logger.info(f"Total images in split dataset: {total_data}")

    logger.info("Dataset split completed successfully!")


if __name__ == "__main__":
    main()
