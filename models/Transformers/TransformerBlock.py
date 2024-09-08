import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )
        self.ffn = tf.keras.models.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(0.1)

    def call(self, x, training):
        attn_output = self.attention(x, x)
        attn_output = self.dropout(attn_output.keras.ut, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerModelBuilder:
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
            inputs = tf.keras.layers.Input(shape=self.input_shape)
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))(inputs)
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))(x)
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))(x)
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))(x)
            x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
            
            # Transformer Encoder
            x = tf.keras.layers.Reshape((-1, x.shape[-1]))(x)
            transformer_block = TransformerBlock(num_heads=4, d_model=x.shape[-1], ff_dim=128)(x)
            x = tf.keras.layers.LSTM(128)(transformer_block)

            outputs = tf.keras.layers.Dense(64, activation='relu')(x)
            outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(outputs)

            model = tf.keras.models.Model(inputs, outputs)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            logging.info("Transformer model successfully built and compiled.")
            return model
        except Exception as e:
            logging.error(f"Error building or compiling the Transformer model: {e}")
            raise

# Example usage
if __name__ == "__main__":
    input_shape = (None, 224, 224, 3)  # Example input shape
    num_classes = 10  # Example number of classes

    try:
        builder = TransformerModelBuilder(input_shape, num_classes)
        transformer_model = builder.build_model()
        transformer_model.summary()
    except ValueError as e:
        logging.error(f"Validation error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")