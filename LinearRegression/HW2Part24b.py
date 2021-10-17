import numpy as np

import pandas as pd
import numpy as np
import math
import json
from numpy import array
from numpy.linalg import norm
import matplotlib.pyplot as plt
import random


def main():
    print('Training Starts Here')
    print()
    train = pd.read_csv('train.csv', header=None)
    df = pd.DataFrame({'Cement': train.loc[:, 0],
                       'Slag': train.loc[:, 1],
                       'Fly_ash': train.loc[:, 2],
                       'Water': train.loc[:, 3],
                       'SP': train.loc[:, 4],
                       'CoarseAggr': train.loc[:, 5],
                       'FineAggr': train.loc[:, 6],
                       'Output: ': train.loc[:, 7]})

    weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    learning_rate = 0.002
    counter = 1

    T_val = []
    cost_function = []
    T = 10000
    for y in range(T):
        x = random.randint(0, df.shape[0]-1)
        print(x)
        a = df.iloc[x, 7]
        b = weights[0] * df.iloc[x, 0]
        c = weights[1] * df.iloc[x, 1]
        d = weights[2] * df.iloc[x, 2]
        e = weights[3] * df.iloc[x, 3]
        f = weights[4] * df.iloc[x, 4]
        g = weights[5] * df.iloc[x, 5]
        h = weights[6] * df.iloc[x, 6]
        bracket = (a - (b + c + d + e + f + g + h))

        weights[0] = weights[0] + learning_rate * bracket * df.iloc[x, 0]
        weights[1] = weights[1] + learning_rate * bracket * df.iloc[x, 1]
        weights[2] = weights[2] + learning_rate * bracket * df.iloc[x, 2]
        weights[3] = weights[3] + learning_rate * bracket * df.iloc[x, 3]
        weights[4] = weights[4] + learning_rate * bracket * df.iloc[x, 4]
        weights[5] = weights[5] + learning_rate * bracket * df.iloc[x, 5]
        weights[6] = weights[6] + learning_rate * bracket * df.iloc[x, 6]
        print('Sample:', counter, 'gives these weights w1: ', weights[0], 'w2: ', weights[1], 'w3: ', weights[2], 'w4: ', weights[3],
              'w5: ', weights[4], 'w6: ', weights[5], 'w7: ', weights[6])
        counter += 1

        sum_value = 0
        for x in range(df.shape[0]):
            value = weights[0] * df.iloc[x, 0] + weights[1] * df.iloc[x, 1] + weights[2] * df.iloc[x, 2] \
                    + weights[3] * df.iloc[x, 3] + weights[4] * df.iloc[x, 4] + weights[5] * df.iloc[x, 5] \
                    + weights[6] * df.iloc[x, 6]
            value = df.iloc[x, 7] - value
            value = value * value
            sum_value += value
        cost_function.append(0.5 * sum_value)
        T_val.append(y)

    plt.plot(T_val, cost_function)
    plt.xlabel('T value')
    plt.ylabel('Cost at this T')
    plt.title('Cost vs T value')
    plt.show()

    print('Testing Starts Here')
    print()
    train = pd.read_csv('test.csv', header=None)
    df1 = pd.DataFrame({'Cement': train.loc[:, 0],
                        'Slag': train.loc[:, 1],
                        'Fly_ash': train.loc[:, 2],
                        'Water': train.loc[:, 3],
                        'SP': train.loc[:, 4],
                        'CoarseAggr': train.loc[:, 5],
                        'FineAggr': train.loc[:, 6],
                        'Output: ': train.loc[:, 7]})

    test_val = 0
    for x in range(df1.shape[0]):
        value = weights[0] * df1.iloc[x, 0] + weights[1] * df1.iloc[x, 1] + weights[2] * df1.iloc[x, 2] \
                + weights[3] * df1.iloc[x, 3] + weights[4] * df1.iloc[x, 4] + weights[5] * df1.iloc[x, 5] \
                + weights[6] * df1.iloc[x, 6]
        value = df1.iloc[x, 7] - value
        value = value * value
        test_val += value

    print(test_val * 0.5)

    # X = np.array([[1, -1, 2, 1],
    #               [1, 1, 3, 1],
    #               [-1, 1, 0, 1],
    #               [1, 2, -4, 1],
    #               [3, -1, -1, 1]])
    #
    # print(X)
    #
    # Y = np.array([
    #     [1],
    #     [4],
    #     [-1],
    #     [-2],
    #     [0]
    # ])
    #
    # w = np.array([0.0, 0.0, 0.0, 0.0])
    #
    # counter = 1
    # for x in range(5):
    #     for y in range(x, x + 1):
    #         # print(w[0])
    #         # print(w[1])
    #         # print(w[2])
    #         # print(w[3])
    #         # print(Y[y])
    #         # print(X[y][0])
    #         # print(X[y][1])
    #         # print(X[y][2])
    #         # print(X[y][3])
    #         temp0 = w[0] + 0.1 * (Y[y] - (w[0] * X[y][0] + w[1] * X[y][1] + w[2] * X[y][2] + w[3] * X[y][3])) * X[y][0]
    #         temp1 = w[1] + 0.1 * (Y[y] - (w[0] * X[y][0] + w[1] * X[y][1] + w[2] * X[y][2] + w[3] * X[y][3])) * X[y][1]
    #         temp2 = w[2] + 0.1 * (Y[y] - (w[0] * X[y][0] + w[1] * X[y][1] + w[2] * X[y][2] + w[3] * X[y][3])) * X[y][2]
    #         temp3 = w[3] + 0.1 * (Y[y] - (w[0] * X[y][0] + w[1] * X[y][1] + w[2] * X[y][2] + w[3] * X[y][3])) * X[y][3]
    #         print('Sample:', counter, 'gives these weights w1: ', temp0, 'w2: ', temp1, 'w3: ', temp2, 'b: ', temp3)
    #         counter += 1
    #
    #     w[0] = temp0
    #     w[1] = temp1
    #     w[2] = temp2
    #     w[3] = temp3
    # print()
    # print('Final weights: ', 'w1: ', w[0], 'w2: ', w[1], 'w3: ', w[2], 'b: ', w[3])
    # for x in range(1, 100):
    #     print(w[0] * x, w[1] * x, w[2] * x, w[3] * x)
    #     print()
    # print(Y)
    #
    # X_transpose = X.transpose()
    #
    # print(X_transpose)
    #
    # XXT = np.dot(X_transpose, X)
    #
    # print(XXT)
    #
    # XXTInverse = np.linalg.inv(XXT)
    #
    # print(XXTInverse)
    #
    # XTY = np.dot(X_transpose, Y)
    #
    # print(XTY)
    #
    # finalproduct = np.dot(XXTInverse, XTY)
    #
    # print(finalproduct)


main()
