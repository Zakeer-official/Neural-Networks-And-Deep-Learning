from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(50, 25, 12), max_iter=600, random_state=42)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

def plot_mlp(mlp):
    # Get the number of layers and their sizes
    layer_sizes = [mlp.n_features_in_] + list(mlp.hidden_layer_sizes) + [mlp.n_outputs_]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot nodes
    for i, layer_size in enumerate(layer_sizes):
        layer_nodes = np.arange(layer_size) + 0.5
        ax.scatter([i] * layer_size, layer_nodes, s=100, zorder=4)
    
    # Plot edges
    for i in range(len(layer_sizes) - 1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i + 1]):
                ax.plot([i, i + 1], [j + 0.5, k + 0.5], 'gray', alpha=0.2)
    
    # Set labels and title
    ax.set_xticks(range(len(layer_sizes)))
    ax.set_xticklabels(['Input'] + [f'Hidden {i+1}' for i in range(len(layer_sizes) - 2)] + ['Output'])
    ax.set_ylabel('Node')
    ax.set_title('Multilayer Perceptron Architecture')
    
    plt.tight_layout()
    plt.show()

plot_mlp(mlp)