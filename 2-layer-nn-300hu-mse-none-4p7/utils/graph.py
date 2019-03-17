import pickle

def load_weights(filepath, neural_network):
    weights_data = pickle.load(filepath)
    neural_network.set_specific_weights(weights_data)

def save_weights(neural_network, filepath):
    weights_data = neural_network.weights
    pickle.dump(weights_data, filepath)


