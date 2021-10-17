import pandas as pd
import numpy as np
import math
import json
from numpy import array
from numpy.linalg import norm
import matplotlib.pyplot as plt


def training_part():
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
    temp_weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    rate = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.008, 0.004, 0.002, 0.001, 0.0001,
            0.00001, 0.000001, 0.0000001, 0.00000005, 0.000000045, 0.00000004, 0.0000000375, 0.000000025, 0.00000001]
    cost_function = []
    t_val = []
    for z in range(101):

        sum_value = 0
        for x in range(df.shape[0]):
            value = weights[0] * df.iloc[x, 0] + weights[1] * df.iloc[x, 1] + weights[2] * df.iloc[x, 2] \
                    + weights[3] * df.iloc[x, 3] + weights[4] * df.iloc[x, 4] + weights[5] * df.iloc[x, 5] \
                    + weights[6] * df.iloc[x, 6]
            value = df.iloc[x, 7] - value
            value = value * value
            sum_value += value
        cost_function.append(0.5 * sum_value)
        t_val.append(z)

        for y in range(df.shape[1] - 1):
            gradient_w = 0
            for x in range(df.shape[0]):
                summation = weights[0] * df.iloc[x, 0] + weights[1] * df.iloc[x, 1] + weights[2] * df.iloc[x, 2] \
                            + weights[3] * df.iloc[x, 3] + weights[4] * df.iloc[x, 4] + weights[5] * df.iloc[x, 5] \
                            + weights[6] * df.iloc[x, 6]
                gradient_w = gradient_w + (df.iloc[x, 7] - summation) * df.iloc[x, y]
            temp_weights[y] = gradient_w * -1

        t_minus_1_weights = weights.copy()
        print('preW: ', t_minus_1_weights)

        print('newW: ', temp_weights)
        for i in range(len(weights)):
            weights[i] = weights[i] - (rate[18] * temp_weights[i])

        t_weights = weights.copy()
        print('finalW: ', t_weights)

        difference_vector = []
        for x in range(len(t_weights)):
            difference_vector.append(t_weights[x] - t_minus_1_weights[x])
        print(difference_vector)

        if norm(difference_vector) < 0.000001:
            print(norm(difference_vector))
            print('yESSSSSSSSSSSSSSSSS')
        else:
            print(norm(difference_vector))
            print('NOOOOOOOOOOOOOOOOOOOOOO')

    print(cost_function)
    print(t_val)

    plt.plot(t_val, cost_function)
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

    print(test_val*0.5)


def main():
    training_part()


main()
