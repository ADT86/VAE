# File: data_loader.py
from tensorflow.keras.datasets import mnist

def load_mnist_data():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return x_train, x_test