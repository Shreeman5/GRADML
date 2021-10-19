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



main()
