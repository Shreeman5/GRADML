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

    X = np.array([[0 for x in range(7)] for y in range(53)]).astype(np.float32)
    print(X)

    Y = np.array([[0 for x in range(1)] for y in range(53)]).astype(np.float32)
    print(Y)

    for x in range(df.shape[0]):
        for y in range(7):
            X[x][y] = df.iloc[x, y]
    print(X)

    for x in range(df.shape[0]):
        for y in range(1):
            Y[x][y] = df.iloc[x, 7]
    print(Y)
    #
    X_transpose = X.transpose()

    print(X_transpose)

    XXT = np.dot(X_transpose, X)

    print(XXT)

    XXTInverse = np.linalg.inv(XXT)

    print(XXTInverse)

    XTY = np.dot(X_transpose, Y)

    print(XTY)

    finalproduct = np.dot(XXTInverse, XTY)

    print(finalproduct)


main()
