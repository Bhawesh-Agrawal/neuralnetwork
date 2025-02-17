import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import keras
from keras import layers

class LinearRegression:
    def __init__(self, data:int = 10000, noise:float = 0.2, random_state:int = 42, test_size:float = 0.2):
        self.data:int = data
        self.noise:float = noise
        self.random_state:int = random_state
        self.test_size:float = test_size
        self.X = 0
        self.y = 0
        self.X_train = 0
        self.X_test = 0
        self.y_train = 0
        self.y_test = 0

    def input_data(self):
        self.X, self.y = make_moons(n_samples=self.data, noise=self.noise, random_state=self.random_state)
        return self.X, self.y
    
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
        return self.X_train, self.X_test, self.y_train, self.y_test, self.X, self.y

    def plot_raw_data(self):
        plt.figure(figsize = (10, 5))
        plt.scatter(self.X[:, 0], self.X[:, 1], c = self.y, cmap='viridis')
        plt.savefig("raw_data.png")
        plt.close()

    def train_model(self, nooflayers:int = 2, noofneurons:list = [10, 10], activation:list = ["relu", "relu"], optimizer:str = "adam", loss:str = "binary_crossentropy", metrics:list = ["accuracy"], epochs:int = 10, validation_split:float = 0.2, batch_size:int = 32):
        model = keras.Sequential()
        for i in range(nooflayers):
            model.add(layers.Dense(noofneurons[i], activation = activation[i]))
        model.add(layers.Dense(1, activation = "sigmoid"))
        model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
        history = model.fit(self.X_train, self.y_train, epochs = epochs, validation_split = validation_split, batch_size = batch_size)
        return history

    def plot_accuracy(self, history):
        plt.figure(figsize = (10, 5))
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.savefig("accuracy.png")
        plt.close()
        
    def plot_loss(self, history):
        plt.figure(figsize = (10, 5))
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.savefig("loss.png")
        plt.close()