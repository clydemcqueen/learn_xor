#!/usr/bin/env python3

import numpy as np


def sigmoid(A):
    """
    Compute the sigmoid activation function for each element in a numpy array.

    σ(a) = 1 / (1 + e^(-a))
    """
    return 1 / (1 + np.exp(-A))


def sigmoid_derivative(S):
    """
    Compute the derivative of the sigmoid function for each element in a numpy array.

    σ'(a) = σ(a) * (1 - σ(a))

    The input must be s = σ(a)
    """
    return S * (1 - S)


class NeuralNetwork:
    """
    A simple feedforward neural network with one hidden layer.

    Numpy allows us to run a set of samples through the network:
        One row per sample, one column per feature
        Use @ for matrix multiplication
        Rely on broadcasting for element-wise addition and multiplication

    Sizes:
        m:          Number of samples (rows in X, H, O)
        input_n:    Number of features in x (columns in X)
        hidden_n:   Number of hidden features (columns in H)
        output_n:   Number of features in y (columns in O)

    Forward pass for a set of samples X:
        Z1 = X @ w1 + b1
        H = σ(Z1)
        Z2 = H @ w2 + b2
        O = σ(Z2)

    Where:
        σ is the element-wise sigmoid function
        X is a set of samples with shape(rows=m, columns=input_n)
        H is the hidden layer output with shape(m, hidden_n)
        O is the predicted output with shape(m, output_n)
        w1 is the weight matrix for the hidden layer with shape(input_n, hidden_n)
        b1 is the bias vector for the hidden layer with shape(1, hidden_n)
        w2 is the weight matrix for the output layer with shape(hidden_n, output_n)
        b2 is the bias vector for the output layer with shape(1, output_n)

    For training and testing we have labels Y with shape(m, output_n).

    Per-sample loss function L is the squared error:
        L = 1/2 * (O - Y)^2
        L' = O - Y

    The overall loss function J is the mean-squared-error, or np.mean(L).

    Back propagation for w2 and b2 uses the chain rule:
        ∂L/∂w2 = ∂L/∂O * ∂O/∂Z2 * ∂Z2/∂w2
        ∂L/∂b2 = ∂L/∂O * ∂O/∂Z2 * ∂Z2/∂b2

    Where:
        ∂L/∂O = O - Y
        ∂O/∂Z2 = O * (1 - O), since O = σ(Z2) from the forward pass
        ∂Z2/∂w2 = H.T
        ∂Z2/∂b2 = 1

    Back propagation for w1 and b1:
        ∂L/∂w1 = ∂L/∂O * ∂O/∂Z2 * ∂Z2/∂H * ∂H/∂Z1 * ∂Z1/∂w1
        ∂L/∂b1 = ∂L/∂O * ∂O/∂Z2 * ∂Z2/∂H * ∂H/∂Z1 * ∂Z1/∂b1

    Where:
        ∂Z2/∂H = w2.T
        ∂H/∂Z1 = H * (1 - H), since H = σ(Z1) from the forward pass
        ∂Z1/∂w1 = X.T
        ∂Z1/∂b1 = 1
    """

    def __init__(self, input_n, hidden_n, output_n):
        """
        Initialize the neural network with random weights and biases.
        """
        self.input_n = input_n
        self.hidden_n = hidden_n
        self.output_n = output_n

        # Initialize weights with random values from standard_normal(mean=0, variance=1)
        self.w1 = np.random.randn(self.input_n, self.hidden_n)
        self.w2 = np.random.randn(self.hidden_n, self.output_n)

        # Initialize biases to 0
        self.b1 = np.zeros((1, self.hidden_n))
        self.b2 = np.zeros((1, self.output_n))

        # Keep the output of each layer for back-propagation
        self.H = None
        self.O = None

    def forward(self, X):
        """
        Run the network forward to make predictions.
        """
        assert len(X.shape) == 2, "X must be a 2D matrix"
        assert X.shape[1] == self.input_n, "Expected " + self.input_n + " features in X"

        self.H = sigmoid(X @ self.w1 + self.b1)

        self.O = sigmoid(self.H @ self.w2 + self.b2)

        return self.O

    def backprop(self, X, Y, learning_rate):
        """
        Use gradient descent to update the weights and biases.
        """
        assert len(X.shape) == 2, "X must be a 2D matrix"
        assert X.shape[1] == self.input_n, "Expected " + self.input_n + " features in X"
        assert Y.shape[1] == self.output_n, "Expected " + self.output_n + " features in Y"
        assert Y.shape[0] == X.shape[0], "Number of samples in X and Y do not match"

        # ∂L/∂O = O - Y
        dL_dO = self.O - Y

        # ∂L/∂Z2 = ∂L/∂O * ∂O/∂Z2
        dL_dZ2 = dL_dO * sigmoid_derivative(self.O)

        # Update w2 using gradient descent
        # ∂L/∂w2 = ∂L/∂O * ∂O/∂Z2 * ∂Z2/∂w2, where  ∂Z2/∂w2 = H.T
        self.w2 -= (self.H.T @ dL_dZ2) * learning_rate

        # Update b2 using gradient descent
        # ∂L/∂b2 = ∂L/∂O * ∂O/∂Z2 * ∂Z2/∂b2, where ∂Z2/∂b2 = 1
        self.b2 -= np.sum(dL_dZ2, axis=0, keepdims=True) * learning_rate

        # ∂L/∂H = ∂L/∂O * ∂O/∂Z2 * ∂Z2/∂H, where ∂Z2/∂H = w2.T
        dL_dH = dL_dZ2 @ self.w2.T

        # ∂L/∂Z1 = ∂L/∂O * ∂O/∂Z2 * ∂Z2/∂H * ∂H/∂Z1
        dL_dZ1 = dL_dH * sigmoid_derivative(self.H)

        # Update w1 using gradient descent
        # ∂L/∂w1 = ∂L/∂O * ∂O/∂Z2 * ∂Z2/∂H * ∂H/∂Z1 * ∂Z1/∂w1, where ∂Z1/∂w1 = X.T
        self.w1 -= (X.T @ dL_dZ1) * learning_rate

        # Update b1 using gradient descent
        # ∂L/∂b1 = ∂L/∂O * ∂O/∂Z2 * ∂Z2/∂H * ∂H/∂Z1 * ∂Z1/∂b1, where ∂Z1/∂b1 = 1
        self.b1 -= np.sum(dL_dZ1, axis=0, keepdims=True) * learning_rate

    def train(self, X, Y, learning_rate, epochs):
        """
        Train the neural network.
        """
        J = None
        for epoch in range(epochs):
            # Make predictions
            O = self.forward(X)

            # Run the back-propagation algorithm
            self.backprop(X, Y, learning_rate)

            # Quit early if we can
            if epoch % 100 == 0:
                L = 0.5 * (Y - O) ** 2
                J = np.mean(L)
                if J < 0.01:
                    print(f"Epoch {epoch}, loss {J:.4f}, we're done!")
                    return

        print(f"Epoch {epochs}, loss {J:.4f}, failed to converge!")


def learn_xor():
    # Training set (and test set) -- each row is a sample
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Labels
    Y = np.array([[0], [1], [1], [0]])

    hidden_n = 3  # Often learns with 2 hidden units, nearly always learns w/ 3 hidden units

    nn = NeuralNetwork(len(X[0]), hidden_n, len(Y[0]))

    # Train the network
    learning_rate = 0.1
    epochs = 10000
    nn.train(X, Y, learning_rate, epochs)

    # Run the network
    predictions = nn.forward(X)

    # Return % correct
    return int(np.mean(np.round(predictions) == Y) * 100)


if __name__ == "__main__":
    results = [learn_xor() for i in range(10)]
    print(f'Results from {len(results)} runs: {results}')
