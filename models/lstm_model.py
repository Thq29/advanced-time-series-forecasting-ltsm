
# Example LSTM model builder using TensorFlow Keras.
def build_lstm(input_shape, units=64):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
    except Exception as e:
        raise e
    model = Sequential([
        InputLayer(input_shape=input_shape),
        LSTM(units, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
