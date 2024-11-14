import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

class Perceptron(nn.Module):
    def __init__(self, n_features):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def activation_func(x):
    return torch.where(x > 0.5, 1.0, 0.0)

def train(model, x_train, y_train, learning_rate, epochs):
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        model.train()
        
        optimizer.zero_grad()
        outputs = model(x_train).squeeze()
        loss = criterion(outputs, y_train)
        
        loss.backward()
        optimizer.step()

def accuracy(y_true, y_pred):
    return (y_true == y_pred).sum().item() / len(y_true)

if __name__ == "__main__":
    X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05)
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = Perceptron(n_features=2)
    
    train(model, X_train, y_train, learning_rate=0.01, epochs=100)
    
    model.eval()
    with torch.no_grad():
        predictions = activation_func(model(X_test).squeeze())

    acc = accuracy(y_test, predictions)
    print(f"Perceptron classification accuracy: {acc:.2f}")
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = torch.amin(X_train[:, 0])
    x0_2 = torch.amax(X_train[:, 0])

    weights = model.linear.weight[0].detach()
    bias = model.linear.bias.detach()

    x1_1 = (-weights[0] * x0_1 - bias) / weights[1]
    x1_2 = (-weights[0] * x0_2 - bias) / weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = torch.amin(X_train[:, 1])
    ymax = torch.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()
