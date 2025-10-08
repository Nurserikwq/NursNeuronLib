import numpy as np
from nnfs.datasets import spiral_data


class NursNeuron:

    inputs = [[1,2,3],[4,5,6],[7,8,9]]
    weights = [0.2, 0.4, 0.3]
    bias = 3
    def neuron(self, inputs, weights, bias):
        self.outputlist = []
        for t in inputs:
            self.output = sum(i * w for i, w in zip(t, weights)) + bias
            self.outputlist.append(self.output)
        return self.outputlist


class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases



class ActivationRelu:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)




X, y = spiral_data(samples = 100, classes = 3)

dense1 = Dense(2, 3)
activation1 = ActivationRelu()
dense1.forward(X)
activation1.forward(dense1.output)

print(activation1.output[:5])