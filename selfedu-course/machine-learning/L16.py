import numpy as np

x_train = np.array([[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]])
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

mw1, ml1 = np.mean(x_train[y_train == 1], axis=0)
mw_1, ml_1 = np.std(x_train[y_train == 1], axis=0)

sw1, sl1 = np.var(x_train[y_train == 1], axis=0, ddof=1)
sw_1, sl_1 = np.var(x_train[y_train == -1], axis=0, ddof=1)

print('MO: ', mw1, ml1, mw_1, ml_1)
print('Dispersions:', sw1, sl1, sw_1, sl_1)

x = [10, 40]

a_1 = lambda x: -np.log(sw_1 * sl_1) - (x[0] - mw_1) ** 2 / (2 * sw_1) - (x[1] - ml_1) ** 2 / (2 * sl_1)
a1 = lambda x: -np.log(sw1 * sl1) - (x[0] - mw1) ** 2 / (2 * sw1) - (x[1] - ml1) ** 2 / (2 * sl1)
y = np.argmax([a_1(x), a1(x)]) * 2 - 1

print('Class number (-1 - caterpillar, 1 - beetle): ', y)

pr = []
for x in x_train:
    pr.append(np.argmax([a_1(x), a1(x)]) * 2 - 1)

pr = np.array(pr)
Q = np.mean(pr != y_train)
print(Q)