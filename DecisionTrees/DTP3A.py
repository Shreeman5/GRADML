import pandas as pd
import numpy as np
import math
import json


def get_subtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)


def ME(df, columns, labels):
    arr = []
    for x in range(len(labels)):
        arr.append(len(df.loc[df['y'] == labels[x]]))
    max_value = max(arr)

    counter = (df.shape[0] - max_value) / df.shape[0]
    ME_values = []
    for (columnName, columnData) in columns.iteritems():
        columnHead = columnName
        values = columnData.values
        total_entropy = 0
        for x in range(len(values)):
            if values[x] != 'Nan':
                arr_1 = []
                a = len(df.loc[df[columnHead] == values[x]])
                if a != 0:
                    b = len(df.loc[df[columnHead] == values[x]].loc[df['y'] == 'yes'])
                    if b == 0:
                        b = a
                    arr_1.append(b)
                    c = len(df.loc[df[columnHead] == values[x]].loc[df['y'] == 'no'])
                    if c == 0:
                        c = a
                    arr_1.append(c)
                    max_value_1 = max(arr_1)
                    total_entropy += (a / df.shape[0]) * (1 - (max_value_1 / a))

        ME_values.append(counter - total_entropy)

    max_value = max(ME_values)
    max_index = ME_values.index(max_value)

    counter_1 = 0
    best_column = ''
    for (columnName, columnData) in columns.iteritems():
        if counter_1 == max_index:
            best_column = columnName
        counter_1 += 1

    return best_column


def makeMETree(df, columns, labels, depth, MEtree=None):
    Class = df.keys()[-1]

    attribute = ME(df, columns, labels)

    attribute_values = np.unique(df[attribute])

    if MEtree is None:
        MEtree = {}
        MEtree[attribute] = {}

    for value in attribute_values:
        subtable = get_subtable(df, attribute, value)
        clValue, counts = np.unique(subtable['y'], return_counts=True)

        if len(counts) == 1:
            MEtree[attribute][value] = clValue[0]
        else:
            if depth > 0:
                MEtree[attribute][value] = makeMETree(subtable, columns.drop(attribute, axis=1),
                                                      labels, depth - 1)

    return MEtree


def GI(df, columns, labels):
    counter = 0

    for x in range(len(labels)):
        counter += (len(df.loc[df['y'] == labels[x]]) / df.shape[0]) ** 2

    counter = 1 - counter

    GI_values = []
    for (columnName, columnData) in columns.iteritems():
        columnHead = columnName
        values = columnData.values
        total_entropy = 0

        for x in range(len(values)):
            if values[x] != 'Nan':
                a = len(df.loc[df[columnHead] == values[x]])
                if a != 0:
                    b = len(df.loc[df[columnHead] == values[x]].loc[df['y'] == 'yes'])
                    c = len(df.loc[df[columnHead] == values[x]].loc[df['y'] == 'no'])

                    total_entropy += a / df.shape[0] * (1 - (b / a) ** 2 - (c / a) ** 2)

        GI_values.append(counter - total_entropy)

    max_value = max(GI_values)
    max_index = GI_values.index(max_value)

    counter_1 = 0
    best_column = ''
    for (columnName, columnData) in columns.iteritems():
        if counter_1 == max_index:
            best_column = columnName
        counter_1 += 1

    return best_column


def makeGITree(df, columns, labels, depth, GITree=None):
    Class = df.keys()[-1]

    attribute = GI(df, columns, labels)

    attribute_values = np.unique(df[attribute])

    if GITree is None:
        GITree = {}
        GITree[attribute] = {}

    for value in attribute_values:
        subtable = get_subtable(df, attribute, value)
        clValue, counts = np.unique(subtable['y'], return_counts=True)

        if len(counts) == 1:
            GITree[attribute][value] = clValue[0]
        else:
            if depth > 0:
                GITree[attribute][value] = makeGITree(subtable, columns.drop(attribute, axis=1),
                                                      labels, depth - 1)

    return GITree


def ID3(df, columns, labels):
    entropy = -1
    counter = 0
    for x in range(len(labels)):
        if len(df.loc[df['y'] == labels[x]]) > 0:
            counter += (len(df.loc[df['y'] == labels[x]]) / df.shape[0]) * (math.log2(
                (len(df.loc[df['y'] == labels[x]]) / df.shape[0])))

    attribute_entropy = []
    for (columnName, columnData) in columns.iteritems():
        columnHead = columnName
        values = columnData.values
        total_entropy = 0
        for x in range(len(values)):
            if values[x] != 'Nan':
                a = len(df.loc[df[columnHead] == values[x]])
                if a != 0:
                    b = len(df.loc[df[columnHead] == values[x]].loc[df['y'] == 'yes'])
                    if b == 0:
                        b = a
                    c = len(df.loc[df[columnHead] == values[x]].loc[df['y'] == 'no'])
                    if c == 0:
                        c = a

                    total_entropy += a / df.shape[0] * -(
                            (b / a * math.log2(b / a)) + (c / a * math.log2(c / a)))

        attribute_entropy.append((entropy * counter) - total_entropy)

    max_value = max(attribute_entropy)
    max_index = attribute_entropy.index(max_value)

    counter_1 = 0
    best_column = ''
    for (columnName, columnData) in columns.iteritems():
        if counter_1 == max_index:
            best_column = columnName
        counter_1 += 1

    return best_column


def makeID3Tree(df, columns, labels, depth, ID3tree=None):
    Class = df.keys()[-1]

    attribute = ID3(df, columns, labels)

    attribute_values = np.unique(df[attribute])

    if ID3tree is None:
        ID3tree = {}
        ID3tree[attribute] = {}

    for value in attribute_values:
        subtable = get_subtable(df, attribute, value)
        clValue, counts = np.unique(subtable['y'], return_counts=True)

        if len(counts) == 1:
            ID3tree[attribute][value] = clValue[0]
        else:
            if depth > 0:
                ID3tree[attribute][value] = makeID3Tree(subtable, columns.drop(attribute, axis=1),
                                                        labels, depth - 1)

    return ID3tree


def get_all_values(nested_dictionary, instance):
    for key, value in nested_dictionary.items():
        if type(value) is dict and len(value) != 0:
            x = value
            y = x.keys()
            for key1 in y:
                if (key, key1) in instance.items():
                    if type(value[key1]) is not dict:
                        return value[key1]
                    else:
                        return get_all_values(value[key1], instance)


def predict(dict_ID3, train):
    array = []
    for x in range(train.shape[0]):
        dict_1 = {'age': train.loc[x, 0], 'job': train.loc[x, 1], 'marital_status': train.loc[x, 2],
                  'education': train.loc[x, 3], 'default_credit': train.loc[x, 4], 'balance': train.loc[x, 5],
                  'housing': train.loc[x, 6], 'loan': train.loc[x, 7], 'contact': train.loc[x, 8],
                  'day': train.loc[x, 9], 'month': train.loc[x, 10], 'duration': train.loc[x, 11],
                  'campaign': train.loc[x, 12], 'pdays': train.loc[x, 13], 'previous': train.loc[x, 14],
                  'poutcome': train.loc[x, 15]
                  }
        val = get_all_values(dict_ID3, dict_1)
        array.append(val)
    return array


def checkAccuracy(arr, train):
    counter = 0
    for x in range(len(arr)):
        a = arr[x]
        b = train.loc[x, 16]
        if a == b:
            counter += 1

    percent_accuracy = ((len(arr) - counter) / len(arr)) * 100
    return percent_accuracy


def find_median(train, column):
    arr_0 = []
    for x in range(train.shape[0]):
        arr_0.append(train.iloc[x, column])

    arr_0.sort()
    mid = len(arr_0) // 2
    res = (arr_0[mid] + arr_0[~mid]) / 2

    return res


def changed_column(train, column, res):
    for x in range(train.shape[0]):
        if train.iloc[x, column] > res:
            train.iloc[x, column] = str(res) + '+'
        else:
            train.iloc[x, column] = str(res) + '-'
    return train.loc[:, column]


def training_part():
    print('Training Starts Here')
    print()

    median_vals_for_test = []
    training_trees = []
    train = pd.read_csv('train.csv', header=None)

    median_val = find_median(train, 0)
    median_vals_for_test.append(median_val)
    column_0 = changed_column(train, 0, median_val)

    median_val = find_median(train, 5)
    median_vals_for_test.append(median_val)
    column_5 = changed_column(train, 5, median_val)

    median_val = find_median(train, 9)
    median_vals_for_test.append(median_val)
    column_9 = changed_column(train, 9, median_val)

    median_val = find_median(train, 11)
    median_vals_for_test.append(median_val)
    column_11 = changed_column(train, 11, median_val)

    median_val = find_median(train, 12)
    median_vals_for_test.append(median_val)
    column_12 = changed_column(train, 12, median_val)

    median_val = find_median(train, 13)
    median_vals_for_test.append(median_val)
    column_13 = changed_column(train, 13, median_val)

    median_val = find_median(train, 14)
    median_vals_for_test.append(median_val)
    column_14 = changed_column(train, 14, median_val)

    df = pd.DataFrame({'age': column_0,
                       'job': train.loc[:, 1],
                       'marital_status': train.loc[:, 2],
                       'education': train.loc[:, 3],
                       'default_credit': train.loc[:, 4],
                       'balance': column_5,
                       'housing': train.loc[:, 6],
                       'loan': train.loc[:, 7],
                       'contact': train.loc[:, 8],
                       'day': column_9,
                       'month': train.loc[:, 10],
                       'duration': column_11,
                       'campaign': column_12,
                       'pdays': column_13,
                       'previous': column_14,
                       'poutcome': train.loc[:, 15],
                       'y': train.loc[:, 16]})

    labels = ['yes', 'no']

    # print(df)

    columns = pd.DataFrame(
        {'age': ['38.0+', '38.0-', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'],
         'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student',
                 'blue-collar', 'self-employed', 'retired', 'technician', 'services'],
         'marital_status': ['married', 'divorced', 'single', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                            'Nan'],
         'education': ['unknown', 'secondary', 'primary', 'tertiary', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                       'Nan'],
         'default_credit': ['yes', 'no', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'],
         'balance': ['452.5+', '452.5-', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'],
         'housing': ['yes', 'no', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'],
         'loan': ['yes', 'no', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'],
         'contact': ['unknown', 'telephone', 'cellular', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'],
         'day': ['16.0+', '16.0-', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'],
         'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
         'duration': ['180.0+', '180.0-', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'],
         'campaign': ['2.0+', '2.0-', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'],
         'pdays': ['-1.0+', '-1.0-', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'],
         'previous': ['0.0+', '0.0-', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'],
         'poutcome': ['unknown', 'other', 'failure', 'success', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan']
         })
    # print(columns)
    #
    depth = 16  # specify depth of dictionary here

    #
    for x in range(depth):
        dict_ID3 = makeID3Tree(df, columns, labels, x)
        ID3_arr = predict(dict_ID3, train)
        ID3_percent = checkAccuracy(ID3_arr, train)
        print('ID3 error at depth', x + 1, 'is:', ID3_percent, '%')

        dict_GI = makeGITree(df, columns, labels, x)
        GI_arr = predict(dict_GI, train)
        GI_percent = checkAccuracy(GI_arr, train)
        print('GI error at depth', x + 1, 'is:', GI_percent, '%')

        dict_ME = makeMETree(df, columns, labels, x)
        ME_arr = predict(dict_ME, train)
        ME_percent = checkAccuracy(ME_arr, train)
        print('ME error at depth', x + 1, 'is:', ME_percent, '%')

        training_trees.append(dict_ID3)
        training_trees.append(dict_GI)
        training_trees.append(dict_ME)

    return training_trees, median_vals_for_test


def testing_part(trees, median_vals):
    print()
    print('Testing Starts Here')
    print()

    print(median_vals)
    test = pd.read_csv('test.csv', header=None)

    test.loc[:, 0] = changed_column(test, 0, median_vals[0])
    test.loc[:, 5] = changed_column(test, 5, median_vals[1])
    test.loc[:, 9] = changed_column(test, 9, median_vals[2])
    test.loc[:, 11] = changed_column(test, 11, median_vals[3])
    test.loc[:, 12] = changed_column(test, 12, median_vals[4])
    test.loc[:, 13] = changed_column(test, 13, median_vals[5])
    test.loc[:, 14] = changed_column(test, 14, median_vals[6])

    print(test)

    depth = 16  # specify depth of dictionary here

    for x in range(depth):
        a = x * 3
        b = a + 1
        c = b + 1
        ID3_arr = predict(trees[a], test)
        GI_arr = predict(trees[b], test)
        ME_arr = predict(trees[c], test)

        ID3_percent = checkAccuracy(ID3_arr, test)
        print('ID3 error at depth', x + 1, 'is:', ID3_percent, '%')
        GI_percent = checkAccuracy(GI_arr, test)
        print('GI error at depth', x + 1, 'is:', GI_percent, '%')
        ME_percent = checkAccuracy(ME_arr, test)
        print('ME error at depth', x + 1, 'is:', ME_percent, '%')


def main():
    trees, median_vals = training_part()
    print()
    print(len(trees))
    testing_part(trees, median_vals)


main()
