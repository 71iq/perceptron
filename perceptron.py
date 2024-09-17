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

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    p = Perceptron()
    p.train(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()













