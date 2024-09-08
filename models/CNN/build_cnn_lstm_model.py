import tensorflow as tf
# from tensorflow.keras import layers, models

def build_cnn_lstm_model(input_shape):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu')))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))

    model.add(tf.keras.layers.LSTM(128, return_sequences=False))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))  # Adjust to the number of classes

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = (None, 224, 224, 3)  # Adjust for time, height, width, and channels
cnn_lstm_model = build_cnn_lstm_model(input_shape)
cnn_lstm_model.summary()
