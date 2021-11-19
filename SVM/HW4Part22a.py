import pandas as pd
import numpy as np
import math
import json
import csv
import matplotlib.pyplot as plt


def training_and_testing_part():
    train = pd.read_csv('train1.csv', header=None)
    train[5] = train[5].map({0: -1, 1: 1})
    weights = np.array([0, 0, 0, 0, 0])
    learning_rate = 0.1
    a_val = 1
    C_value = 700 / 873

    for x in range(100):
        df = train.sample(frac=1).reset_index(drop=True)
        learning_rate = learning_rate / (1 + ((learning_rate * x)/a_val))

        for y in range(df.shape[0]):
            a = weights[0] * df.iloc[y, 0]
            b = weights[1] * df.iloc[y, 1]
            c = weights[2] * df.iloc[y, 2]
            d = weights[3] * df.iloc[y, 3]
            e = weights[4] * df.iloc[y, 4]

            y_prime = a + b + c + d + e

            if y_prime * df.iloc[y, 5] <= 1:
                copy_weights = []
                for z in range(len(weights) - 1):
                    copy_weights.append(weights[z] * learning_rate)
                copy_weights.append(0)
                second_term_scalar = learning_rate * C_value * (df.shape[0]) * df.iloc[y, 5]
                second_term_vector = [second_term_scalar * df.iloc[y, 0], second_term_scalar * df.iloc[y, 1],
                                      second_term_scalar * df.iloc[y, 2], second_term_scalar * df.iloc[y, 3],
                                      second_term_scalar * df.iloc[y, 4]]
                first_2_terms = np.subtract(weights, copy_weights)
                weights = np.add(first_2_terms, second_term_vector)
            else:
                for z in range(len(weights) - 1):
                    weights[z] = weights[z] * (1 - learning_rate)

    print(weights)

    train_error_counter = 0
    for y in range(train.shape[0]):
        f = weights[0] * train.iloc[y, 0]
        g = weights[1] * train.iloc[y, 1]
        h = weights[2] * train.iloc[y, 2]
        i = weights[3] * train.iloc[y, 3]
        j = weights[4] * train.iloc[y, 4]

        y_prime = f + g + h + i + j

        if y_prime <= 0:
            y_prime = -1
        else:
            y_prime = 1

        if y_prime != train.iloc[y, 5]:
            train_error_counter = train_error_counter + 1

    print(train_error_counter)

    test = pd.read_csv('test1.csv', header=None)
    test[5] = test[5].map({0: -1, 1: 1})
    print(test)
    test_error_counter = 0
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
            test_error_counter = test_error_counter + 1

    print(test_error_counter)

    return (train_error_counter / train.shape[0]) * 100, (test_error_counter / test.shape[0]) * 100


def main():
    training_error, testing_error = training_and_testing_part()
    print(training_error)
    print(testing_error)
    # testing_part(h_t_arr, vote, median_vals_for_test)


main()
