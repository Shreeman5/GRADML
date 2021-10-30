import pandas as pd
import numpy as np
import math
import json
import csv


def training_and_testing_part():
    train = pd.read_csv('train.csv', header=None)
    train[5] = train[5].map({0: -1, 1: 1})
    print(train)
    weights = np.array([0, 0, 0, 0, 0])

    for x in range(10):
        df = train.sample(frac=1).reset_index(drop=True)
        print(df)

        for y in range(df.shape[0]):
            a = weights[0] * df.iloc[y, 0]
            b = weights[1] * df.iloc[y, 1]
            c = weights[2] * df.iloc[y, 2]
            d = weights[3] * df.iloc[y, 3]
            e = weights[4] * df.iloc[y, 4]

            y_prime = a + b + c + d + e

            if y_prime <= 0:
                y_prime = -1
            else:
                y_prime = 1

            if y_prime != df.iloc[y, 5]:
                print('yeah')
                v = np.array([df.iloc[y, 0], df.iloc[y, 1], df.iloc[y, 2], df.iloc[y, 3], df.iloc[y, 4]])
                w = df.iloc[y, 5] * v
                value = 0.25 * w
                weights = np.add(weights, value)
                print(weights)
            else:
                print('no')
                print(weights)

            print()

    test = pd.read_csv('test.csv', header=None)
    test[5] = test[5].map({0: -1, 1: 1})
    print(test)
    counter = 0
    for y in range(test.shape[0]):
        f = weights[0] * test.iloc[y, 0]
        g = weights[1] * test.iloc[y, 1]
        h = weights[2] * test.iloc[y, 2]
        i = weights[3] * test.iloc[y, 3]
        j = weights[4] * test.iloc[y, 4]

        y_prime = f + g + h + i + j

        if y_prime <= 0:
            y_prime = -1
        else:
            y_prime = 1

        if y_prime != test.iloc[y, 5]:
            counter = counter + 1

    print(counter)

    return (counter / test.shape[0]) * 100


def main():
    error = training_and_testing_part()
    print(error)
    # testing_part(h_t_arr, vote, median_vals_for_test)


main()
