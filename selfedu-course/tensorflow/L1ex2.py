import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

x = tf.Variable(-1.0)
y = lambda: x ** 2 - x

N = 100
opt = tf.optimizers.SGD(learning_rate=0.1)
for n in range(N):
    opt.minimize(y, [x])

print(x.numpy())