from tensorflow import keras


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

