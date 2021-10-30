import pandas as pd
import numpy as np
import math
import json
import csv


def training_and_testing_part():
    df = pd.read_csv('train.csv', header=None)
    df[5] = df[5].map({0: -1, 1: 1})
    print(df)
    weights = np.array([0, 0, 0, 0, 0])
    counter = 0
    combo_weights = []
    counter_weights = []

    for x in range(10):
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
                combo_weights.append(weights)
                counter_weights.append(counter)
                print('yeah')
                v = np.array([df.iloc[y, 0], df.iloc[y, 1], df.iloc[y, 2], df.iloc[y, 3], df.iloc[y, 4]])
                w = df.iloc[y, 5] * v
                value = 0.75 * w
                weights = np.add(weights, value)
                counter = 1
                print(weights)
            else:
                counter = counter + 1
                print('no')
                print(weights)

            print()

    combo_weights.append(weights)
    counter_weights.append(counter)

    print(combo_weights)
    print(counter_weights)
    test = pd.read_csv('test.csv', header=None)
    test[5] = test[5].map({0: -1, 1: 1})
    print(test)

    error = 0
    for y in range(test.shape[0]):

        counter = 0
        for z in range(len(combo_weights)):
            weight = combo_weights[z]

            f = weight[0] * test.iloc[y, 0]
            g = weight[1] * test.iloc[y, 1]
            h = weight[2] * test.iloc[y, 2]
            i = weight[3] * test.iloc[y, 3]
            j = weight[4] * test.iloc[y, 4]

            y_prime = f + g + h + i + j

            if y_prime <= 0:
                y_prime = -1
            else:
                y_prime = 1

            value = counter_weights[z] * y_prime
            counter = counter + value

        if counter <= 0:
            counter = -1
        else:
            counter = 1

        if counter != test.iloc[y, 5]:
            error = error + 1

    print(error)

    row = []
    for x in range(len(combo_weights)):
        row.append([x, combo_weights[x], counter_weights[x]])
    fields = ['k', 'weight at k', 'count for this weight']
    with open("sample_submit2.csv", 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(fields)
        csv_writer.writerows(row)
    #
    return (error / test.shape[0]) * 100


def main():
    error = training_and_testing_part()
    print(error)
    # testing_part(h_t_arr, vote, median_vals_for_test)


main()
