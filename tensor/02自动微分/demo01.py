import tensorflow as tf
import numpy as np

#在autograph中完成 f(x) = ax^2 + bx + c 的最小值求解
# 使用optimizer.minimize

x = tf.Variable(np.random.randint(0, 10), dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# 无参数放在优化器中
@tf.function
def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    # x^2 - 2x + b
    y = a * tf.pow(x, 2) + b* x + c
    return y

@tf.function
def train(epoch):
    for batch in tf.range(epoch):
        optimizer.minimize(f, [x])
    return f()

tf.print(train(1000))
tf.print(x)

