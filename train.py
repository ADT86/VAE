# File: train.py
from tensorflow.keras.optimizers import Adam

def train_model(model, data, epochs=10, batch_size=32):
    model.compile(optimizer=Adam(), loss='mse')
    model.fit(data, data, epochs=epochs, batch_size=batch_size)