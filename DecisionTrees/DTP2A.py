import pandas as pd
import numpy as np
import math


def get_subtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)


def ME(df, columns, labels):
    arr = []
    for x in range(len(labels)):
        arr.append(len(df.loc[df['label'] == labels[x]]))
    max_value = max(arr)

    counter = (df.shape[0] - max_value) / df.shape[0]
    ME_values = []
    for (columnName, columnData) in columns.iteritems():
        columnHead = columnName
        values = columnData.values
        total_entropy = 0
        for x in range(len(values)):
            if values[x] != 'NaN':
                arr_1 = []
                a = len(df.loc[df[columnHead] == values[x]])
                if a != 0:
                    b = len(df.loc[df[columnHead] == values[x]].loc[df['label'] == 'unacc'])
                    arr_1.append(b)
                    if b == 0:
                        b = a
                    c = len(df.loc[df[columnHead] == values[x]].loc[df['label'] == 'acc'])
                    arr_1.append(c)
                    if c == 0:
                        c = a
                    d = len(df.loc[df[columnHead] == values[x]].loc[df['label'] == 'good'])
                    arr_1.append(d)
                    if d == 0:
                        d = a
                    e = len(df.loc[df[columnHead] == values[x]].loc[df['label'] == 'vgood'])
                    arr_1.append(e)
                    if e == 0:
                        e = a
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
        clValue, counts = np.unique(subtable['label'], return_counts=True)

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
        counter += (len(df.loc[df['label'] == labels[x]]) / df.shape[0]) ** 2

    counter = 1 - counter

    GI_values = []
    for (columnName, columnData) in columns.iteritems():
        columnHead = columnName
        values = columnData.values
        total_entropy = 0

        for x in range(len(values)):
            if values[x] != 'NaN':
                a = len(df.loc[df[columnHead] == values[x]])
                if a != 0:
                    b = len(df.loc[df[columnHead] == values[x]].loc[df['label'] == 'unacc'])
                    c = len(df.loc[df[columnHead] == values[x]].loc[df['label'] == 'acc'])
                    d = len(df.loc[df[columnHead] == values[x]].loc[df['label'] == 'good'])
                    e = len(df.loc[df[columnHead] == values[x]].loc[df['label'] == 'vgood'])

                    total_entropy += a / df.shape[0] * (1 - (b / a) ** 2 - (c / a) ** 2 - (d / a) ** 2 - (e / a) ** 2)

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
        clValue, counts = np.unique(subtable['label'], return_counts=True)

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
        if len(df.loc[df['label'] == labels[x]]) > 0:
            counter += (len(df.loc[df['label'] == labels[x]]) / df.shape[0]) * (math.log2(
                (len(df.loc[df['label'] == labels[x]]) / df.shape[0])))
    attribute_entropy = []
    for (columnName, columnData) in columns.iteritems():
        columnHead = columnName
        values = columnData.values
        total_entropy = 0
        for x in range(len(values)):
            if values[x] != 'NaN':
                a = len(df.loc[df[columnHead] == values[x]])
                if a != 0:
                    b = len(df.loc[df[columnHead] == values[x]].loc[df['label'] == 'unacc'])
                    if b == 0:
                        b = a
                    c = len(df.loc[df[columnHead] == values[x]].loc[df['label'] == 'acc'])
                    if c == 0:
                        c = a
                    d = len(df.loc[df[columnHead] == values[x]].loc[df['label'] == 'good'])
                    if d == 0:
                        d = a
                    e = len(df.loc[df[columnHead] == values[x]].loc[df['label'] == 'vgood'])
                    if e == 0:
                        e = a

                    total_entropy += a / df.shape[0] * -(
                            (b / a * math.log2(b / a)) + (c / a * math.log2(c / a)) + (d / a * math.log2(d / a)) + (
                            e / a * math.log2(e / a)))

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
        clValue, counts = np.unique(subtable['label'], return_counts=True)

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
        dict_1 = {'buying': train.loc[x, 0], 'maint': train.loc[x, 1], 'doors': train.loc[x, 2],
                  'persons': train.loc[x, 3],
                  'lug_boot': train.loc[x, 4], 'safety': train.loc[x, 5]}
        val = get_all_values(dict_ID3, dict_1)
        array.append(val)

    return array


def checkAccuracy(arr, train):
    counter = 0
    for x in range(len(arr)):
        a = arr[x]
        b = train.loc[x, 6]
        if a == b:
            counter += 1

    percent_accuracy = ((len(arr) - counter) / len(arr)) * 100
    return percent_accuracy


def training_part():
    print('Training Starts Here')
    print()

    training_trees = []
    train = pd.read_csv('train1.csv', header=None)

    df = pd.DataFrame({'buying': train.loc[:, 0],
                       'maint': train.loc[:, 1],
                       'doors': train.loc[:, 2],
                       'persons': train.loc[:, 3],
                       'lug_boot': train.loc[:, 4],
                       'safety': train.loc[:, 5],
                       'label': train.loc[:, 6]})
    labels = ['unacc', 'acc', 'good', 'vgood']

    columns = pd.DataFrame({'buying': ['vhigh', 'high', 'med', 'low'],
                            'maint': ['vhigh', 'high', 'med', 'low'],
                            'doors': ['2', '3', '4', '5more'],
                            'persons': ['2', '4', 'more', 'NaN'],
                            'lug_boot': ['small', 'med', 'big', 'NaN'],
                            'safety': ['low', 'med', 'high', 'NaN']})

    depth = 6  # specify depth of dictionary here

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

    return training_trees


def testing_part(trees):
    print()
    print('Testing Starts Here')
    print()

    test = pd.read_csv('test1.csv', header=None)

    depth = 6  # specify depth of dictionary here

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
    trees = training_part()
    testing_part(trees)


main()
