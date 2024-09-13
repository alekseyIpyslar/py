# Adaboost Regression Algorithm on Decision Trees

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

np.random.seed(123)

x = np.arange(0, np.pi/2, 0.1).reshape(-1, 1)
y = np.sin(x) + np.random.normal(0, 0.1, x.shape)

# plt.plot(x, y)
# plt.grid()
# plt.show()

T = 5          # number of algorithms in the composition
max_depth = 2  # maximum depth of decision trees
algs = []      # list of obtained algorithms
s = np.array(y.ravel())
for n in range(T):
    # create and train a decisive tree
    algs.append(DecisionTreeRegressor(max_depth=max_depth))
    algs[-1].fit(x, s)

    s -= algs[-1].predict(x)  # recalculate residuals

# reconstruct the original signal from the set of obtained trees
yy = algs[0].predict(x)
for n in range(1, T):
    yy += algs[n].predict(x)

# display results as plots
plt.plot(x, y)      # original plot
plt.plot(x, yy)     # reconstructed plot
plt.plot(x, s)      # residual signal
plt.grid()
plt.show()