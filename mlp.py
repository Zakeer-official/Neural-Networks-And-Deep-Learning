import numpy as np
import json

class MLP:
    def __init__(self, config_path):
        # Load configuration from JSON file
        with open(config_path, 'r') as file:
            config = json.load(file)
        
        self.input_size = config['input_size']
        self.layers_config = config['layers']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.initialize_network()
    
    def initialize_network(self):
        self.weights = []
        self.biases = []
        layer_input_size = self.input_size
        
        for layer in self.layers_config:
            layer_size = layer['size']
            self.weights.append(np.random.randn(layer_input_size, layer_size))
            self.biases.append(np.random.randn(1, layer_size))
            layer_input_size = layer_size

    def activation(self, x, func):
        if func == 'relu':
            return np.maximum(0, x)
        elif func == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif func == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError(f"Unknown activation function: {func}")
    
    def activation_derivative(self, x, func):
        if func == 'relu':
            return np.where(x > 0, 1, 0)
        elif func == 'sigmoid':
            return x * (1 - x)
        elif func == 'tanh':
            return 1 - x ** 2
        else:
            raise ValueError(f"Unknown activation function: {func}")
    
    def forward_propagation(self, X):
        self.z = []
        self.a = [X]
        for i, layer in enumerate(self.layers_config):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.z.append(z)
            a = self.activation(z, layer['activation'])
            self.a.append(a)
        return self.a[-1]
    
    def backward_propagation(self, X, y):
        m = y.shape[0]
        delta = self.a[-1] - y
        d_weights = []
        d_biases = []

        for i in reversed(range(len(self.weights))):
            d_weights.insert(0, np.dot(self.a[i].T, delta) / m)
            d_biases.insert(0, np.sum(delta, axis=0, keepdims=True) / m)
            if i != 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.a[i], self.layers_config[i-1]['activation'])
        
        return d_weights, d_biases
    
    def update_weights(self, d_weights, d_biases):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * d_weights[i]
            self.biases[i] -= self.learning_rate * d_biases[i]
    
    def train(self, X, y):
        for epoch in range(self.epochs):
            output = self.forward_propagation(X)
            d_weights, d_biases = self.backward_propagation(X, y)
            self.update_weights(d_weights, d_biases)
            if epoch % 100 == 0:
                loss = np.mean((output - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss}")
    
    def predict(self, X):
        return self.forward_propagation(X)

# Example usage:
if __name__ == "__main__":
    config_path = 'mlp.json'
    
    X = np.array([[0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 1, 1]])  # Example input
    y = np.array([[0], [1], [1], [0]])  # Example output

    mlp = MLP(config_path)
    mlp.train(X, y)
    predictions = mlp.predict(X)
    print("Predictions:", predictions)
