import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.weights = None
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_func(self, x):
        return np.where(x > 0, 1, 0)

    def train(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.random.random(n_features)
        y_ = np.where(y > 0, 1, 0)

        for _ in range(self.epochs):
            for idx, x_i in enumerate(x):
                prediction = self.predict(x_i)
                update = self.learning_rate * (y_[idx] - prediction)
                self.weights += update * x_i
                self.bias += update

    def predict(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        return self.activation_func(linear_output)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import numpy as np

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_blobs(
        n_samples=150, n_features=3, centers=2, cluster_std=1.05, random_state=2
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    p = Perceptron()
    p.train(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, marker='o')

    x0_range = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 10)
    x1_range = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 10)
    x0_grid, x1_grid = np.meshgrid(x0_range, x1_range)
    x2_grid = (-p.weights[0] * x0_grid - p.weights[1] * x1_grid - p.bias) / p.weights[2]

    ax.plot_surface(x0_grid, x1_grid, x2_grid, color='k', alpha=0.3)

    plt.show()
