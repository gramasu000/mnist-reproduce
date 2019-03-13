import numpy as np

class NeuralNetwork:

    def __init__(self):
        self.input_size = 784
        self.hidden_size = 300
        self.output_size = 10

    def set_random_weights(self):
        self.weights = {
            "w1": np.random.randn((self.input_size, self.hidden_size)),
            "w2": np.random.randn((self.hidden_size, self.output_size)),
            "b1": np.random.randn((1, self.hidden_size))
            "b2": np.random.randn((1, self.output_size))
        }
    
    def set_specific_weights(self, weights):
        self.weights = weights

    @staticmethod
    def learning_rate(iter):
        return 0.001

    @staticmethod
    def sofm(x):
        s = np.exp(x) 
        return s / np.sum(s)
    
    @staticmethod
    def relu(x):
        return np.greater(x, 0) * x

    def _forward_prop(self, input):
        h1 = np.matmul(input, self.weights["w1"]) + self.weights["b1"]
        r1 = self.relu(h1)
        h2 = np.matmul(r1, self.weights["w2"]) + self.weights["b2"]
        probs = self.sofm(h2)
        return h1, r1, h2, probs
    
    def predict(self, input):
        _,_,_,probs = self._forward_prop(input)
        preds = np.argmax(probs, axis=1)
        return preds

    def _compute_loss(self, probs, labels):
        loss = np.sum(-np.log(probs) * labels)
        return loss

    def _compute_gradients(self, h1, r1, h2, probs):
        # TODO: Finish this
