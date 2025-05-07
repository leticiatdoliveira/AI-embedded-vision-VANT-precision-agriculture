import tensorflow as tf
import os

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
        
    @staticmethod
    def preprocess_for_mobilenet(image, label):
        """Preprocess images specifically for MobileNet."""
        # MobileNet preprocessing expects values in [-1, 1]
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

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
        
    def load_mobilenet_datasets(self):
        """Load and prepare datasets with MobileNet-specific preprocessing."""
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

        # Transform datasets with MobileNet-specific preprocessing
        train_ds = train_ds.map(self.preprocess_for_mobilenet).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.map(self.preprocess_for_mobilenet).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = test_ds.map(self.preprocess_for_mobilenet).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

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

