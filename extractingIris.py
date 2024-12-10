import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from perceptron import Perceptron
from matplotlib.colors import ListedColormap

s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print ('From URL:', s)

df = pd.read_csv(s, header=None, encoding='utf-8')
# print(df.tail())

#select setosa and versicolor
y = df.iloc[0:100, 4].values # so basically here I am extracting the labels which might be Iris-setos or iris-versicolor
print(df)
print('--------------------------------------------------------------------')
print(y)

y = np.where(y == 'Iris-setosa', 0, 1) # here is a condition if it is iris-setosa it will give a 0 otherwis 1 but I will get an array
print('--------------------------------------------------------------------')
print(y)
# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values # here is extracting just two properties from each row the first and the third [0, 2]
print('--------------------------------------------------------------------')
print(X)
# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red' , marker='o', label='Setosa') # the first parameters seems to be the x and y axis wich his value will comes from the sepal length and the petal lenght
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue' , marker='s', label='Versicolor')
# here is simply labeling those making them available for the chart that will launch later...


plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron(eta=0.1,  n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_)+ 1),
         ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

def plot_decision_regions(X, y, classifier, resolution = 0.02):
    # setup marker generator and color map 
    markes = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[len(np.unique(y))])

    #plot the decision surpface
    x1_min, x1_max = X[X:, 0].min() -1, X[:,0].max() +1
    x2_min, x2_max = X[X:, 1].min() -1, X[:,1].max() +1

    xx1, xx2=np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                         np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

