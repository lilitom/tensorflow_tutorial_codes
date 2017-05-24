from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import mean_squared_error
from math import sqrt
from functools import reduce
import random
import numpy as np


def random_sum_pair(n_examples, n_numbers, largest, op):
    """生成数字对
    """
    X, y = [], []
    for i in range(n_examples):
        in_pattern = [random.randint(0, largest) for _ in range(n_numbers)]
        out_pattern = reduce(op, in_pattern)
        X.append(in_pattern)
        y.append(out_pattern)
    X, y = np.array(X), np.array(y)
    # normalize?
    X = X.astype('float') / float(n_numbers * largest)
    y = y.astype('float') / float(n_numbers * largest)
    return X, y


def convert(val, n_numbers, largest):
    return np.round(val * (n_numbers * largest))


n_examples = 100
n_numbers = 2
largest = 100
n_epochs = 100

model = Sequential()
model.add(Dense(units=4, input_shape=(2,)))
model.add(Dense(units=2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse')

# train model
for _ in range(n_epochs):
    X, y = random_sum_pair(n_examples, n_numbers, largest, op=lambda x, y: x - y)
    model.fit(X, y, epochs=1, verbose=2)

X_test, y_test = random_sum_pair(n_examples, n_numbers, largest, op=lambda x, y: x - y)
y_pred = model.predict(X_test)[:, 0]

# convert
y_test = convert(y_test, n_numbers, largest)
y_pred = convert(y_pred, n_numbers, largest)

# RMSE
rmse = sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))
print('RMSE:{}'.format(rmse))

# print some prediction
for i in range(20):
    print('Expected:{}, Predicted:{} (Err={})'.format(y_test[i], y_pred[i], y_test[i] - y_pred[i]))
