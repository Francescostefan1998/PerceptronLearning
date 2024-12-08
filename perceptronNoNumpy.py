import random

class PerceptronNoNumpy:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        random.seed(self.random_state)
        self.w_ = [random.uniform(-0.01, 0.01) for _ in range(len(X[0]))]  # Initialize weights randomly.
        self.b_ = 0.0  # Initialize bias to zero.
        self.errors_ = []

        for _ in range(self.n_iter):  # Loop over epochs.
            errors = 0
            for xi, target in zip(X, y):  # Loop over all training examples.
                update = self.eta * (target - self.predict(xi))  # Calculate weight update.
                self.w_ = [w + update * x for w, x in zip(self.w_, xi)]  # Update weights.
                self.b_ += update  # Update bias.
                errors += int(update != 0.0)  # Count misclassifications.
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return sum(w * x for w, x in zip(self.w_, X)) + self.b_

    def predict(self, X):
        return 1 if self.net_input(X) >= 0.0 else 0
