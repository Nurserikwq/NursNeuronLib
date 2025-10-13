import numpy as np
from nnfs.datasets import spiral_data
import math

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


class ActivationSoftMax:

    def foward(self, inputs):

        self.exp_values - np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = self.exp_values / np.sum(self.exp_values, axis=1, keepdims=True)
        self.output = probabilities



class Loss:
    def calculate (self, output, y):

        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)
        return data_loss

    def Loss_CategoricalCrossentropy(Loss):

        def forward(self, y_pred, y_true):
            samples = len(y_pred)
            y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

            if len(y_true.shape) == 1:
                correct_confidences = y_pred_clipped[
                    range(samples),
                    y_true
                ]
            elif len(y_true.shape) == 2:
                correct_confidences = np.sum(
                    y_pred_clipped * y_true,
                    axis=1
                )
            # Losses
            negative_log_likelihoods = -np.log(correct_confidences)
            return negative_log_likelihoods


    def acc (self, y_pred, y_true):
        def forward(self, y_pred, y_true):
            predictions = np.argmax(y_pred, axis=1)
            if len(y_true.shape) == 2:
                class_targets = np.argmax(y_true, axis=1)
            accuracy = np.mean(predictions == class_targets)

            return(accuracy)





