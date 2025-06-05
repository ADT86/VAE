# File: model.py
from tensorflow.keras import layers, models

def create_vae(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(128, activation='relu')(x)
    z_mean = layers.Dense(2, name='z_mean')(x)
    z_log_var = layers.Dense(2, name='z_log_var')(x)
    return models.Model(inputs, [z_mean, z_log_var], name="vae")