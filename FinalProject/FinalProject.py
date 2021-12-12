import pandas as pd
import numpy as np
import math
import json
import csv


def get_subtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)


# Gini Index method to find the best attribute at every depth
def GI(df, columns, labels):
    counter = 0

    for x in range(len(labels)):
        counter += (len(df.loc[df['income>50K'] == labels[x]]) / df.shape[0]) ** 2

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
                    b = len(df.loc[df[columnHead] == values[x]].loc[df['income>50K'] == '1'])
                    c = len(df.loc[df[columnHead] == values[x]].loc[df['income>50K'] == '0'])

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


# Method that grows the Gini Index decision tree until depth = d
def makeGITree(df, columns, labels, depth, GITree=None):
    if depth == 0:
        positive = len(df.loc[df['income>50K'] == '1'])
        negative = len(df.loc[df['income>50K'] == '0'])
        if negative > positive:
            return '0'
        else:
            return '1'

    Class = df.keys()[-1]

    attribute = GI(df, columns, labels)

    attribute_values = np.unique(df[attribute])

    if GITree is None:
        GITree = {}
        GITree[attribute] = {}

    for value in attribute_values:
        subtable = get_subtable(df, attribute, value)
        clValue, counts = np.unique(subtable['income>50K'], return_counts=True)

        if len(counts) == 1:
            GITree[attribute][value] = clValue[0]
        else:
            GITree[attribute][value] = makeGITree(subtable, columns.drop(attribute, axis=1),
                                                  labels, depth - 1)

    return GITree


# ID3 method to find the best attribute at every depth
def ID3(df, columns, labels):
    entropy = -1
    counter = 0
    for x in range(len(labels)):
        if len(df.loc[df['income>50K'] == labels[x]]) > 0:
            counter += (len(df.loc[df['income>50K'] == labels[x]]) / df.shape[0]) * (math.log2(
                (len(df.loc[df['income>50K'] == labels[x]]) / df.shape[0])))

    attribute_entropy = []
    for (columnName, columnData) in columns.iteritems():
        columnHead = columnName
        values = columnData.values
        total_entropy = 0
        for x in range(len(values)):
            if values[x] != 'Nan':
                a = len(df.loc[df[columnHead] == values[x]])
                if a != 0:
                    b = len(df.loc[df[columnHead] == values[x]].loc[df['income>50K'] == '1'])
                    if b == 0:
                        b = a
                    c = len(df.loc[df[columnHead] == values[x]].loc[df['income>50K'] == '0'])
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


# Method that grows the ID3 decision tree until depth = d
def makeID3Tree(df, columns, labels, depth, ID3tree=None):
    if depth == 0:
        positive = len(df.loc[df['income>50K'] == '1'])
        negative = len(df.loc[df['income>50K'] == '0'])
        if negative > positive:
            return '0'
        else:
            return '1'

    Class = df.keys()[-1]

    attribute = ID3(df, columns, labels)

    attribute_values = np.unique(df[attribute])

    if ID3tree is None:
        ID3tree = {}
        ID3tree[attribute] = {}

    for value in attribute_values:
        subtable = get_subtable(df, attribute, value)
        clValue, counts = np.unique(subtable['income>50K'], return_counts=True)

        if len(counts) == 1:
            ID3tree[attribute][value] = clValue[0]
        else:
            ID3tree[attribute][value] = makeID3Tree(subtable, columns.drop(attribute, axis=1),
                                                    labels, depth - 1)

    return ID3tree


# Use by the predict function to predict one training/testing example
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


# Predicts each training example and returns an array of predictions back to the training function
def predict(dict_ID3, train):
    array = []
    for x in range(0, train.shape[0]):
        dict_1 = {'age': train.iloc[x, 0], 'workclass': train.iloc[x, 1], 'fnlwgt': train.iloc[x, 2],
                  'education': train.iloc[x, 3], 'education.num': train.iloc[x, 4], 'marital.status': train.iloc[x, 5],
                  'occupation': train.iloc[x, 6], 'relationship': train.iloc[x, 7], 'race': train.iloc[x, 8],
                  'sex': train.iloc[x, 9], 'capital.gain': train.iloc[x, 10], 'capital.loss': train.iloc[x, 11],
                  'hours.per.week': train.iloc[x, 12], 'native.country': train.iloc[x, 13]
                  }
        val = get_all_values(dict_ID3, dict_1)
        array.append(val)
    return array


# Predicts each testing example and returns an array of predictions back to the testing function
def predict1(tree, test):
    array = []
    for x in range(0, test.shape[0]):
        dict_1 = {'age': test.iloc[x, 0], 'workclass': test.iloc[x, 1], 'fnlwgt': test.iloc[x, 2],
                  'education': test.iloc[x, 3], 'education.num': test.iloc[x, 4], 'marital.status': test.iloc[x, 5],
                  'occupation': test.iloc[x, 6], 'relationship': test.iloc[x, 7], 'race': test.iloc[x, 8],
                  'sex': test.iloc[x, 9], 'capital.gain': test.iloc[x, 10], 'capital.loss': test.iloc[x, 11],
                  'hours.per.week': test.iloc[x, 12], 'native.country': test.iloc[x, 13]}
        val = get_all_values(tree, dict_1)
        array.append(val)

    return array


# Checks the accuracy of the predictions for the training examples
def checkAccuracy(arr, train):
    counter = 0
    for x in range(len(arr)):
        a = arr[x]
        b = train.iloc[x, 14]
        if a == b:
            counter += 1

    print(counter)
    percent_accuracy = (counter / len(arr)) * 100
    return percent_accuracy


# Method that changes values of attributes age and fnlwgt according to the 7 medians
def changed_column_with_7_medians(train, column, median_val):
    for x in range(1, train.shape[0]):
        value = train.iloc[x, column]
        if median_val[0] < float(value) <= median_val[1]:
            train.iloc[x, column] = str(median_val[0]) + '+'
        elif median_val[1] < float(value) <= median_val[2]:
            train.iloc[x, column] = str(median_val[1]) + '+'
        elif median_val[2] < float(value) <= median_val[3]:
            train.iloc[x, column] = str(median_val[2]) + '+'
        elif median_val[3] < float(value) <= median_val[4]:
            train.iloc[x, column] = str(median_val[3]) + '+'
        elif median_val[4] < float(value) <= median_val[5]:
            train.iloc[x, column] = str(median_val[4]) + '+'
        elif median_val[5] < float(value) <= median_val[6]:
            train.iloc[x, column] = str(median_val[5]) + '+'
        elif median_val[6] < float(value) <= median_val[7]:
            train.iloc[x, column] = str(median_val[6]) + '+'
        elif median_val[7] < float(value) <= median_val[8]:
            train.iloc[x, column] = str(median_val[7]) + '+'


# Method that changes values of attribute education.num according to the 3 medians
def changed_column_with_3_medians(train, column, median_val):
    for x in range(1, train.shape[0]):
        value = train.iloc[x, column]
        if median_val[0] < float(value) <= median_val[1]:
            train.iloc[x, column] = str(median_val[0]) + '+'
        elif median_val[1] < float(value) <= median_val[2]:
            train.iloc[x, column] = str(median_val[1]) + '+'
        elif median_val[2] < float(value) <= median_val[3]:
            train.iloc[x, column] = str(median_val[2]) + '+'
        elif median_val[3] < float(value) <= median_val[4]:
            train.iloc[x, column] = str(median_val[3]) + '+'


# Method that changes values of attributes capital.gain and capital.loss according to the 1 median
def changed_column_with_1_median(train, column):
    for x in range(1, train.shape[0]):
        value = train.iloc[x, column]
        if float(value) <= 0:
            train.iloc[x, column] = str(0) + '-'
        else:
            train.iloc[x, column] = str(0) + '+'


# Method that changes values of attribute hours.per.week according to the 1 median
def changed_column_with_three_sets_based_on_one_value(train, column):
    for x in range(1, train.shape[0]):
        value = train.iloc[x, column]
        if float(value) < 40:
            train.iloc[x, column] = str(40) + '-'
        elif float(value) == 40:
            train.iloc[x, column] = str(40) + '|'
        else:
            train.iloc[x, column] = str(40) + '+'


# Method that replaces the '?' value for any attribute with the majority value of that attribute
def specify_majority_in_column(train, column, string):
    for x in range(1, train.shape[0]):
        if train.iloc[x, column] == '?':
            train.iloc[x, column] = string


# Pre-processing of the training data
def pre_processing_data_train(train):
    median_val_0 = [0, 23.0, 28.0, 33.0, 37.0, 42.0, 48.0, 56.0, 90.0]
    changed_column_with_7_medians(train, 0, median_val_0)

    specify_majority_in_column(train, 1, 'Private')

    median_val_2 = [0, 78410.0, 117210.0, 151158.0, 177304.0, 200194.0, 235661.0, 306868.0, 1490400.0]
    changed_column_with_7_medians(train, 2, median_val_2)

    median_val_4 = [0, 9.0, 10.0, 12.0, 16.0]
    changed_column_with_3_medians(train, 4, median_val_4)

    specify_majority_in_column(train, 6, 'Prof-specialty')

    changed_column_with_1_median(train, 10)

    changed_column_with_1_median(train, 11)

    changed_column_with_three_sets_based_on_one_value(train, 12)

    specify_majority_in_column(train, 13, 'United-States')


# Pre-processing of the testing data
def pre_processing_data_test(test):
    median_val_1 = [0, 23.0, 28.0, 33.0, 37.0, 42.0, 48.0, 56.0, 90.0]
    changed_column_with_7_medians(test, 1, median_val_1)

    specify_majority_in_column(test, 2, 'Private')

    median_val_3 = [0, 78410.0, 117210.0, 151158.0, 177304.0, 200194.0, 235661.0, 306868.0, 1490400.0]
    changed_column_with_7_medians(test, 3, median_val_3)

    median_val_5 = [0, 9.0, 10.0, 12.0, 16.0]
    changed_column_with_3_medians(test, 5, median_val_5)

    specify_majority_in_column(test, 7, 'Prof-specialty')

    changed_column_with_1_median(test, 11)

    changed_column_with_1_median(test, 12)

    changed_column_with_three_sets_based_on_one_value(test, 13)

    specify_majority_in_column(test, 14, 'United-States')


# Splits the data into 5 parts in accordance with cross-validation
def cross_validation(df):
    df_1 = df.iloc[:20000, :]
    df_2 = df.iloc[20000:25000, :]

    df_a = df.iloc[:15000, :]
    df_b = df.iloc[20000:25000, :]
    frames = [df_a, df_b]
    df_3 = pd.concat(frames)
    df_4 = df.iloc[15000:20000, :]

    df_c = df.iloc[:10000, :]
    df_d = df.iloc[15000:25000, :]
    frames = [df_c, df_d]
    df_5 = pd.concat(frames)
    df_6 = df.iloc[10000:15000, :]

    df_e = df.iloc[:5000, :]
    df_f = df.iloc[10000:25000, :]
    frames = [df_e, df_f]
    df_7 = pd.concat(frames)
    df_8 = df.iloc[5000:10000, :]

    df_9 = df.iloc[5000:25000, :]
    df_10 = df.iloc[:5000, :]

    df_values = [df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10]

    return df_values


# find best depth using cross validation
def find_best_depth(df_matrices, columns, labels):
    depth = 14
    for x in range(depth - 1, depth):
        avg_accuracy = 0
        for y in np.arange(0, 10, 2):
            df_to_be_trained = df_matrices[y]
            dict_ID3 = makeID3Tree(df_to_be_trained, columns, labels, depth)
            print('Tree:', dict_ID3)
            df_to_be_tested = df_matrices[y + 1]
            print(df_to_be_tested)
            ID3_arr = predict(dict_ID3, df_to_be_tested)
            print('Length of predictions:', len(ID3_arr))
            ID3_percent = checkAccuracy(ID3_arr, df_to_be_tested)
            print('Accuracy:', ID3_percent)
            avg_accuracy = avg_accuracy + ID3_percent
            # dict_GI = makeGITree(df_to_be_trained, columns, labels, depth)
            # print('Tree:', dict_GI)
            # df_to_be_tested = df_matrices[y + 1]
            # print(df_to_be_tested)
            # GI_arr = predict(dict_GI, df_to_be_tested)
            # print('Length of predictions:', len(GI_arr))
            # GI_percent = checkAccuracy(GI_arr, df_to_be_tested)
            # print('Accuracy:', GI_percent)
            # avg_accuracy = avg_accuracy + GI_percent
        print('Avg accuracy:', avg_accuracy / 5)


# training of the data to give the best hyper parameter, ie depth, for the decision tree to be used in the testing data
# Used for both ID3 and Gini Index
def training_part():
    print('Training Starts Here')
    print()

    train = pd.read_csv('train_final.csv', header=None)

    pre_processing_data_train(train)

    df = pd.DataFrame({'age': train.loc[1:, 0],
                       'workclass': train.loc[1:, 1],
                       'fnlwgt': train.loc[1:, 2],
                       'education': train.loc[1:, 3],
                       'education.num': train.loc[1:, 4],
                       'marital.status': train.loc[1:, 5],
                       'occupation': train.loc[1:, 6],
                       'relationship': train.loc[1:, 7],
                       'race': train.loc[1:, 8],
                       'sex': train.loc[1:, 9],
                       'capital.gain': train.loc[1:, 10],
                       'capital.loss': train.loc[1:, 11],
                       'hours.per.week': train.loc[1:, 12],
                       'native.country': train.loc[1:, 13],
                       'income>50K': train.loc[1:, 14]})

    labels = ['1', '0']

    columns = pd.DataFrame(
        {'age': ['0+', '23.0+', '28.0+', '33.0+', '37.0+', '42.0+', '48.0+', '56.0+', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'],
         'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov',
                       'Without-pay', 'Never-worked', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                       'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                       'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'],
         'fnlwgt': ['0+', '78410.0+', '117210.0+', '151158.0+', '177304.0+', '200194.0+', '235661.0+', '306868.0+',
                    'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                    'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                    'Nan', 'Nan', 'Nan', 'Nan', 'Nan'],
         'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',
                       '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool', 'Nan',
                       'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                       'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'],
         'education.num': ['0+', '9.0+', '10.0+', '12.0+', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                           'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                           'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                           'Nan', 'Nan', 'Nan'],
         'marital.status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
                            'Married-spouse-absent', 'Married-AF-spouse', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                            'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                            'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                            'Nan', 'Nan'],
         'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
                        'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
                        'Priv-house-serv', 'Protective-serv', 'Armed-Forces', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                        'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                        'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'],
         'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', 'Nan', 'Nan',
                          'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                          'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                          'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'],
         'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black', 'Nan', 'Nan', 'Nan', 'Nan',
                  'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                  'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                  'Nan', 'Nan', 'Nan', 'Nan'],
         'sex': ['Female', 'Male', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan'],
         'capital.gain': ['0+', '0-', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                          'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                          'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                          'Nan', 'Nan'],
         'capital.loss': ['0+', '0-', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                          'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                          'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                          'Nan', 'Nan'],
         'hours.per.week': ['40+', '40-', '40|', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                            'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                            'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan', 'Nan',
                            'Nan', 'Nan'],
         'native.country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
                            'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran',
                            'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal',
                            'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia',
                            'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador',
                            'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'],
         })

    # df_matrices = cross_validation(df)
    #
    # find_best_depth(df_matrices, columns, labels)

    best_ID3_tree = makeID3Tree(df, columns, labels, 8)
    best_GI_tree = makeGITree(df, columns, labels, 2)
    print(best_ID3_tree)
    print(best_GI_tree)

    return best_ID3_tree, best_GI_tree


# Predicts results for both Gini and ID3 using Kaggle's testing data
def testing_part(best_ID3_tree, best_GI_tree):
    print()
    print('Testing Starts Here')
    print()

    test = pd.read_csv('test_final.csv', header=None)
    pre_processing_data_test(test)

    test = test.drop(0, axis=1)

    test = test.drop(0, axis=0)

    # print(test)

    ID3_arr = predict1(best_ID3_tree, test)
    for x in range(len(ID3_arr)):
        if ID3_arr[x] != '0' and ID3_arr[x] != '1':
            ID3_arr[x] = '0'

    GI_arr = predict1(best_GI_tree, test)
    for x in range(len(GI_arr)):
        if GI_arr[x] != '0' and GI_arr[x] != '1':
            GI_arr[x] = '0'

    # row = []
    # for x in range(len(GI_arr)):
    #     row.append([x + 1, GI_arr[x]])
    # fields = ['ID', 'Prediction']
    # with open("sample_submit1.csv", 'w') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow(fields)
    #     csv_writer.writerows(row)

    row = []
    for x in range(len(ID3_arr)):
        row.append([x + 1, ID3_arr[x]])
    fields = ['ID', 'Prediction']
    with open("sample_submit1.csv", 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(fields)
        csv_writer.writerows(row)


def main():
    best_ID3_tree, best_GI_tree = training_part()
    testing_part(best_ID3_tree, best_GI_tree)


main()

#
#
# def find_3_medians(train, column):
#     arr_0 = []
#     for x in range(1, train.shape[0]):
#         arr_0.append(float(train.iloc[x, column]))
#
#     arr_0.sort()
#
#     array_length = len(arr_0)
#     two_n_eight = arr_0[math.floor(2 * array_length / 8)]
#     four_n_eight = arr_0[math.floor(4 * array_length / 8)]
#     six_n_eight = arr_0[math.floor(6 * array_length / 8)]
#
#     res = [0, two_n_eight, four_n_eight, six_n_eight, arr_0[array_length - 1]]
#
#     return res


# def specify_majority(train, column):
#     counter_0 = 0
#     for y in range(1, train.shape[0]):
#         if train.iloc[y, column] == 'United-States':
#             counter_0 += 1
#     print(counter_0)

# def find_7_medians(train, column):
#     arr_0 = []
#     for x in range(1, train.shape[0]):
#         arr_0.append(float(train.iloc[x, column]))
#
#     arr_0.sort()
#
#     array_length = len(arr_0)
#     one_n_eight = arr_0[math.floor(array_length / 8)]
#     two_n_eight = arr_0[math.floor(2 * array_length / 8)]
#     three_n_eight = arr_0[math.floor(3 * array_length / 8)]
#     four_n_eight = arr_0[math.floor(4 * array_length / 8)]
#     five_n_eight = arr_0[math.floor(5 * array_length / 8)]
#     six_n_eight = arr_0[math.floor(6 * array_length / 8)]
#     seven_n_eight = arr_0[math.floor(7 * array_length / 8)]
#
#     res = [0, one_n_eight, two_n_eight, three_n_eight, four_n_eight, five_n_eight, six_n_eight, seven_n_eight,
#            arr_0[array_length - 1]]
#
#     return res
