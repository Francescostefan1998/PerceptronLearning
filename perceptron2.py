import numpy as np

class Perceptron2:
    def __init__(self, eta=0.1, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(X.shape[1])
        self.b_ = 0
        self.errors_ = []

        for epoch in range(self.n_iter):
            print(f"Epoch {epoch+1}")
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                print(f"Sample {xi}, Target: {target}, Predict : {self.predict(xi)},  Update: {update}")
                print(f"Initial weight : {self.w_}")
                self.w_ += update * xi
                print(f"Updated weight : {self.w_}")
                print(f"Initial bias : {self.b_}")
                self.b_ += update
                print(f"Updated bias : {self.b_}")
                errors += int(update != 0.0)
                print(f"Weights: {self.w_}, Bias: {self.b_}")
            self.errors_.append(errors)
            print(f"Errors of Epoch : {epoch + 1} : {errors} ")
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
