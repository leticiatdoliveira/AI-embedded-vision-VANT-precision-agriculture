import os
import matplotlib.pyplot as plt
from tensorflow import keras
from callbacks import AccuracyThresholdCallback

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

        return [checkpoint_callback], checkpoint_path

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

