from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from math import sqrt
import random
import numpy as np

random.seed(1)


def random_sum_paris(n_samples, n_numbers, largest):
    X, y = [], []
    for i in range(n_samples):
        in_pattern = [random.randint(0, largest) for _ in range(n_numbers)]
        out_pattern = sum(in_pattern)
        X.append(in_pattern)
        y.append(out_pattern)
    X, y = np.array(X), np.array(y)
    # normalize
    X = X.astype('float') / float(largest * n_numbers)
    y = y.astype('float') / float(largest * n_numbers)
    return X, y


def invert(value, n_numbers, largets):
    # invert normalization
    return round(value * float(largets * n_numbers))


n_examples = 100
n_numbers = 2
largest = 100
n_epochs = 1000
n_batch = 20

# create LSTM
model = Sequential()
model.add(LSTM(units=6, input_shape=(n_numbers, 1), return_sequences=True))
model.add(LSTM(units=6))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse')
# train LSTM
for _ in range(n_epochs):
    X, y = random_sum_paris(n_examples, n_numbers, largest)
    X = X.reshape(n_examples, n_numbers, 1)
    model.fit(X, y, epochs=1, batch_size=n_batch, verbose=2)

# evaluate
X, y = random_sum_paris(n_examples, n_numbers, largest)
X = X.reshape(n_examples, n_numbers, 1)
result = model.predict(X, batch_size=n_batch, verbose=0)
print(result)
# calculate error
expected = [invert(x, n_numbers, largest) for x in y]
predicted = [invert(x, n_numbers, largest) for x in result[:, 0]]

# root mean squared error
rmse = sqrt(mean_squared_error(y_true=expected, y_pred=predicted))
print('RMSE: %f' % rmse)

for i in range(100):
    error = expected[i] - predicted[i]
    print('Exptected=%d, Predicted=%d, (err=%d)' % (expected[i], predicted[i], error))
