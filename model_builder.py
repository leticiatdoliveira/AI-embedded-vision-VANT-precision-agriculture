import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot

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
        
    def build_mobilenet_model(self, num_classes):
        """Build a MobileNetV2-based model with transfer learning."""
        try:
            # Load the MobileNetV2 model with pre-trained weights
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(self.config.IMG_SIZE[0], self.config.IMG_SIZE[1], 3),
                include_top=False,
                weights='imagenet'
            )
            
            # Freeze the base model layers
            base_model.trainable = False
            
            # Create a new model using Functional API
            inputs = tf.keras.Input(shape=(self.config.IMG_SIZE[0], self.config.IMG_SIZE[1], 3))
            
            # Apply data preprocessing required by MobileNetV2
            x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
            
            # Pass the inputs through the base model
            x = base_model(x)
            
            # Add classification head
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(128, activation='relu')(x)
            outputs = layers.Dense(num_classes, activation='softmax')(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            self.logger.info("MobileNet model built successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Error building MobileNet model: {str(e)}")
            raise
        
    @staticmethod
    def create_quantized_model(model):
        """Create a quantization-aware model from an existing model."""
        return tfmot.quantization.keras.quantize_model(model)

    def compile_model(self, model, learning_rate=None):
        """Compile model with specified optimizer and metrics."""
        if learning_rate is None:
            learning_rate = self.config.LEARNING_RATE

        # Use the new Keras 3 optimizer API instead of legacy
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fine_tune_mobilenet_model(self, model, num_classes, fine_tune_at=100):
        """Fine-tune a pre-trained MobileNet model by unfreezing some layers."""
        # Unfreeze the top layers of the MobileNet model
        base_model = model.layers[0]
        base_model.trainable = True
        
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
        # Recompile the model with a lower learning rate for fine-tuning
        model = self.compile_model(model, learning_rate=self.config.LEARNING_RATE / 10)
        
        return model

