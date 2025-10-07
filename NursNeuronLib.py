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



calc = NursNeuron()
print(calc.neuron(calc.inputs, calc.weights, calc.bias))
