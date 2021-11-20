import numpy as np
from scipy.optimize import minimize
import pandas as pd


def objective(alphas, y_product, x_product):
    alpha_t = alphas.T
    alphas_prod = np.matmul(alphas, alpha_t)

    first_prod = np.multiply(y_product, x_product)
    second_prod = np.multiply(first_prod, alphas_prod)
    # print('F:', second_prod.shape)

    first_term = 0.5 * np.sum(second_prod)
    # print('sum: ', first_term)
    second_term = np.sum(alphas)
    # print('s2:', second_term)

    return first_term - second_term


def constrain(y, alphas_i):
    y_w = y.reshape([-1, 1])
    constrains = 0
    for ww in range(872):
        constrains = constrains + (alphas_i[ww] * y_w[ww])
    return constrains


train = pd.read_csv('train3.csv', header=None)
train[5] = train[5].map({0: -1, 1: 1})

alphas = np.zeros(872).reshape([-1, 1])
# print(alphas.shape)
alphas.fill(0.05)
# print(alphas)

x = np.array(train.iloc[:, 0:5])
x_t = x.transpose()

y = np.array(train.iloc[:, 5]).reshape([-1, 1])
# print(y.shape)
y_t = y.T

x_prod = np.matmul(x, x_t)
# print('B:', x_prod.shape)
# print(x_prod)

y_prod = np.matmul(y, y_t)
# print('A:', y_prod.shape)
# print(y_prod)

cons = ({'type': 'eq', 'fun': constrain, "args": (y,)})

bnds = []
for z in range(872):
    bnds.append((0, 700 / 873))

sol = minimize(fun=objective, x0=alphas, args=(y_prod, x_prod), method='SlSQP', bounds=bnds, constraints=cons)
# print(sol.x)
print(sol)

weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

for s in range(872):
    weights[0] = weights[0] + (sol.x[s] * train.iloc[s, 5] * train.iloc[s, 0])
    weights[1] = weights[1] + (sol.x[s] * train.iloc[s, 5] * train.iloc[s, 1])
    weights[2] = weights[2] + (sol.x[s] * train.iloc[s, 5] * train.iloc[s, 2])
    weights[3] = weights[3] + (sol.x[s] * train.iloc[s, 5] * train.iloc[s, 3])

# print(weights)

val = 0
for t in range(872):
    term = (weights[0] * train.iloc[t, 0]) + (weights[1] * train.iloc[t, 1]) + (weights[2] * train.iloc[t, 2]) + (
            weights[3] * train.iloc[t, 3])
    val = val + (train.iloc[t, 5] - term)

weights[4] = val / 872

print(weights)

train_error_counter = 0
for q in range(train.shape[0]):
    f = weights[0] * train.iloc[q, 0]
    g = weights[1] * train.iloc[q, 1]
    h = weights[2] * train.iloc[q, 2]
    i = weights[3] * train.iloc[q, 3]
    j = weights[4] * train.iloc[q, 4]

    y_prime = f + g + h + i + j

    if y_prime <= 0:
        y_prime = -1
    else:
        y_prime = 1

    if y_prime != train.iloc[q, 5]:
        train_error_counter = train_error_counter + 1

print((train_error_counter / train.shape[0]) * 100)

test = pd.read_csv('test3.csv', header=None)
test[5] = test[5].map({0: -1, 1: 1})
print(test)
test_error_counter = 0
for r in range(test.shape[0]):
    f = weights[0] * test.iloc[r, 0]
    g = weights[1] * test.iloc[r, 1]
    h = weights[2] * test.iloc[r, 2]
    i = weights[3] * test.iloc[r, 3]
    j = weights[4] * test.iloc[r, 4]

    y_prime = f + g + h + i + j

    if y_prime <= 0:
        y_prime = -1
    else:
        y_prime = 1

    if y_prime != test.iloc[r, 5]:
        test_error_counter = test_error_counter + 1

print((test_error_counter / test.shape[0]) * 100)
