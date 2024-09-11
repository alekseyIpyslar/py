import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

r1 = 0.8
D1 = 1.0
mean1 = [0, -3]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.7
D2 = 2.0
mean2 = [0, 3]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

Py1, L1 = 0.5, 1
Py2, L2 = 1 - Py1, 1

b = lambda x, v, m, l, py: (np.log(1 * py) - 0.5 * (x - m) @ np.linalg.inv(v)
                               @ (x - m).T - 0.5 * np.log(np.linalg.det(v)))

