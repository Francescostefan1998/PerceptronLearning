import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from perceptron2 import Perceptron2

# Input: [Experience (years), Education Level]
X = np.array([
    [1, 0],  # Little experience, no degree    0
    [3, 1],  # Some experience, bachelor       0
    [5, 2],  # Solid experience, master        1
    [10, 0], # Expert, no degree               1
    [0, 2],  # No experience, master           0
    [7, 1],  # Experienced, bachelor           1
    [2, 1],  # Low experience, bachelor        0
    [8, 0],  # High experience, no degree      1
    [6, 2],  # Decent experience, master       1
    [4, 0],  # Mid experience, no degree       0
    [9, 2],  # Very experienced, master        1
    [1, 2],  # Little experience, master       0
    [3, 0],  # Some experience, no degree      0
    [5, 1],  # Solid experience, bachelor      1
    [8, 2]   # High experience, master         1
])

# Output: 1 = Hired, 0 = Not Hired
y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1])


plt.scatter(X[y == 0][:, 1], X[y == 0][:, 0],  color='red', marker='o', label='Not Hired')
plt.scatter(X[y == 1][:, 1], X[y == 1][:, 0],  color='blue', marker='s', label='Hired')
plt.ylabel('Experience (years)')
plt.xlabel('Education Level')
plt.legend()
plt.title('Candidate Distribution: Hired vs Not Hired')
plt.show()


ppn = Perceptron2(eta=0.1, n_iter=9)
ppn.fit(X, y)

# Visualize the training process
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('Training Errors Over Time')
plt.show()

# Example candidate: 6 years of experience, bachelor's degree (education level = 1)
new_candidate = np.array([6, 1])

# Predict
result = ppn.predict(new_candidate)
print(f"The candidate {'gets hired' if result == 1 else 'does not get hired'}.")