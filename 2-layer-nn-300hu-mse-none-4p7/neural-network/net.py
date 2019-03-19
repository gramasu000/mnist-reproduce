import numpy as np

class NeuralNetwork:

    def __init__(self):
        self.input_size = 784
        self.hidden_size = 300
        self.output_size = 10

    def set_random_weights(self):
        self.weights = {
            "w1": np.random.randn((self.input_size, self.hidden_size)),
            "b1": np.random.randn((1, self.hidden_size)),
            "w2": np.random.randn((self.hidden_size, self.output_size)),
            "b2": np.random.randn((1, self.output_size))
        }
    
    def set_specific_weights(self, weights):
        self.weights = weights

    def learning_rate(iter):
        return 0.001

    def _forward_prop(self, input):  
        h1 = np.matmul(input, self.weights["w1"]) + self.weights["b1"]
        a1 = np.greater(h1, 0) * h1
        h2 = np.matmul(r1, self.weights["w2"]) + self.weights["b2"]
        a2 = np.exp(h2)
        a2 /= np.sum(a2)
        self.fprop = {
            "in": input
            "h1": h1,
            "a1": a1,
            "h2": h2,
            "a2": a2 
        }
    
    def predict(self, input):
        self._forward_prop(input)
        preds = np.argmax(self.fprop["a2"], axis=1)
        return preds

    def _compute_loss(self, probs, labels):
        loss = np.sum(-np.log(probs) * labels)
        return loss

    def _compute_gradients(self, labels):
        dh2 = self.fprop["a2"] - labels
        dw2 = np.matmul(self.fprop["a1"].T, dh2)
        db2 = dh2 
        da1 = np.matmul(dh2, self.weights["w2"].T)
        dh1 = da1 * np.greater(self.fprop["h1"], 0)
        dw1 = np.matmul(self.fprop["in"].T, dh1)
        db1 = dh1 
        self.bprop = {
            "dw2": dw2,
            "db2": db2,
            "dw1": dw1,
            "db1": db1
        }

    def _evaluate_gradients(self, iter):
        for key in self.weights:
            bkey = "d" + key
            self.weights[key] -= self.learning_rate(iter)*self.bprop[bkey]
 
    def _flush(self):
        self.fprop = {}
        self.bprop = {}
