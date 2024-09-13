import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1)
x_est = np.arange(0, 10, 0.01)
N = len(x)
y_sin = np.sin(x)
y = y_sin + np.random.normal(0, 0.5, N)

h = 1.0

# Gaussian kernel
K = lambda r: np.exp(-2 * r * r)
# Triangular kernel
# K = lambda r: np.abs(1 - r) * bool(r <= 1)
# Rectangular kernel
# K = lambda r: bool(r <= 1)

ro = lambda xx, xi: np.abs(xx - xi)
w = lambda xx, xi: K(ro(xx, xi) / h)

plt.figure(figsize=(7, 7))
plot_number = 0

for h in [0.1, 0.3, 1, 10]:
    y_est = []
    for xx in x_est:
        ww = np.array([w(xx, xi) for xi in x])
        yy = np.dot(ww, y) / sum(ww)
        y_est.append(yy)

    plot_number += 1
    plt.subplot(2, 2, plot_number)

    plt.scatter(x, y, c='black', s=10)
    plt.plot(x, y_sin, c='blue')
    plt.plot(x_est, y_est, c='red')
    plt.title(f"Gaussian kernel with h = {h}")
    plt.grid()

plt.show()
