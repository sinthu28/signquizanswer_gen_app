import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CNNLSTMModelBuilder:
    def __init__(self, input_shape, num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.validate_parameters()

    def validate_parameters(self):
        if not isinstance(self.input_shape, tuple) or len(self.input_shape) != 4:
            raise ValueError("Input shape must be a tuple of length 4: (time_steps, height, width, channels).")
        if not isinstance(self.num_classes, int) or self.num_classes <= 0:
            raise ValueError("Number of classes must be a positive integer.")

    def build_model(self):
        try:
            model = tf.keras.models.Sequential()
            
            # CNN layers
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), input_shape=self.input_shape))
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))))
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu')))
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))))
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
            
            # LSTM layers
            model.add(tf.keras.layers.LSTM(128, return_sequences=False))
            model.add(tf.keras.layers.Dense(64, activation='relu'))
            model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
            
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            logging.info("CNN-LSTM model successfully built and compiled.")
            return model
        except Exception as e:
            logging.error(f"Error building or compiling the CNN-LSTM model: {e}")
            raise

        ######################################################################################
"""
        if __name__ == "__main__":
            input_shape = (None, 224, 224, 3)  # Example input shape
            num_classes = 10  # Example number of classes

            try:
                builder = CNNLSTMModelBuilder(input_shape, num_classes)
                cnn_lstm_model = builder.build_model()
                cnn_lstm_model.summary()
            except ValueError as e:
                logging.error(f"Validation error: {e}")
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
"""