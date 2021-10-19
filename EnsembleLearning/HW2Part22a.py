import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt


def get_subtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)


def positive_process_dataframe(df):
    positive_val = 0
    for y in range(df.shape[0]):
        positive_val = positive_val + df.iloc[y, 16]
    return positive_val


def negative_process_dataframe(df):
    negative_val = 0
    for y in range(df.shape[0]):
        negative_val = negative_val + df.iloc[y, 16]
    return negative_val


def a_dataframe(df):
    a_val = 0
    for y in range(df.shape[0]):
        a_val = a_val + df.iloc[y, 16]
    return a_val


def ID3(df, columns, labels):
    # print(df)
    # print(df.loc[:, 16])
    entropy = -1
    counter = 0

    positive_proportion = 0
    negative_proportion = 0
    counter1 = 0
    counter2 = 0
    for x in range(df.shape[0]):
        if df.iloc[x, 17] == 'yes':
            positive_proportion = positive_proportion + df.iloc[x, 16]
            counter1 += 1
        if df.iloc[x, 17] == 'no':
            negative_proportion = negative_proportion + df.iloc[x, 16]
            counter2 += 1
    #
    counter = (positive_proportion * math.log2(positive_proportion) + negative_proportion * math.log2(
        negative_proportion))
    # print('E1:', counter * -1)

    # positive_proportion = positive_proportion * 5000
    # negative_proportion = negative_proportion * 5000
    # counter += (positive_proportion/df.shape[0] * math.log2(positive_proportion/df.shape[0])) + (
    #         negative_proportion/df.shape[0] * math.log2(negative_proportion/df.shape[0]))
    # print('E2:', counter*-1)

    # print('PS:', positive_proportion)
    # print('NS:', negative_proportion)
    # print('Counter1: ', counter1)
    # print('Counter2: ', counter2)
    # print('Entropy:', counter)
    attribute_entropy = []
    for (columnName, columnData) in columns.iteritems():
        columnHead = columnName
        # print('CH:', columnHead)
        values = columnData.values
        total_entropy = 0
        for x in range(len(values)):
            positive_val = 0
            negative_val = 0
            if values[x] != 'Nan':
                # print(values[x])
                a_val = a_dataframe(df.loc[df[columnHead] == values[x]])
                positive_val = positive_process_dataframe(df.loc[df[columnHead] == values[x]].loc[df['y'] == 'yes'])
                negative_val = negative_process_dataframe(df.loc[df[columnHead] == values[x]].loc[df['y'] == 'no'])

                # print('PV:', positive_val)
                # print('NV:', negative_val)
                # a = len(df.loc[df[columnHead] == values[x]])
                # print(a)
                # positive_val = positive_val * df.shape[0]
                # negative_val = negative_val * df.shape[0]
                if a_val != 0:
                    if positive_val == 0:
                        positive_val = a_val
                    if negative_val == 0:
                        negative_val = a_val

                    total_entropy += a_val * -(positive_val / a_val * math.log2(positive_val / a_val)
                                               + negative_val / a_val * math.log2(negative_val / a_val))
                    # print('TE:', total_entropy)

        # print('TEE: ', total_entropy)
        # print('val:', (entropy * counter) - total_entropy)
        attribute_entropy.append((entropy * counter) - total_entropy)

    max_value = max(attribute_entropy)
    max_index = attribute_entropy.index(max_value)
    # print(max_value)
    # print(max_index)

    counter_1 = 0
    best_column = ''
    for (columnName, columnData) in columns.iteritems():
        if counter_1 == max_index:
            best_column = columnName
        counter_1 += 1
    # print('Best Co:', best_column)
    return best_column


def makeID3Tree(df, columns, labels, depth, ID3tree=None):
    if depth == 0:
        # positive = len(df.loc[df['y'] == 'yes'])
        # negative = len(df.loc[df['y'] == 'no'])

        positive = positive_process_dataframe(df.loc[df['y'] == 'yes'])
        negative = negative_process_dataframe(df.loc[df['y'] == 'no'])
        # print('POS: ', positive)
        # print('NEG: ', negative)
        if negative > positive:
            return 'no'
        else:
            return 'yes'

    Class = df.keys()[-1]

    attribute = ID3(df, columns, labels)
    # print(attribute)

    attribute_values = np.unique(df[attribute])
    # print(attribute_values)

    if ID3tree is None:
        ID3tree = {}
        ID3tree[attribute] = {}

    for value in attribute_values:
        subtable = get_subtable(df, attribute, value)
        clValue, counts = np.unique(subtable['y'], return_counts=True)
        # print('Attribute: ', attribute)
        # print('Value: ', value)

        if len(counts) == 1:
            ID3tree[attribute][value] = clValue[0]
        else:
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


def find_weighted_classification_error(arr, df):
    summation = 0
    for x in range(len(arr)):
        a = arr[x]
        b = df.iloc[x, 17]
        if a == b:
            summation += df.iloc[x, 16] * 1
        else:
            summation += df.iloc[x, 16] * -1

    weighted_classification_error = 0.5 - 0.5 * summation
    return weighted_classification_error * 100


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


def compute_error(ID3_arr, df):
    counter = 0
    for x in range(len(ID3_arr)):
        a = ID3_arr[x]
        b = df.iloc[x, 17]
        if a == b:
            counter += 1
    return (5000 - counter) / 5000


def combined_predict(h_t_arr, vote, train):
    array = []
    for x in range(train.shape[0]):
        dict_1 = {'age': train.loc[x, 0], 'job': train.loc[x, 1], 'marital_status': train.loc[x, 2],
                  'education': train.loc[x, 3], 'default_credit': train.loc[x, 4], 'balance': train.loc[x, 5],
                  'housing': train.loc[x, 6], 'loan': train.loc[x, 7], 'contact': train.loc[x, 8],
                  'day': train.loc[x, 9], 'month': train.loc[x, 10], 'duration': train.loc[x, 11],
                  'campaign': train.loc[x, 12], 'pdays': train.loc[x, 13], 'previous': train.loc[x, 14],
                  'poutcome': train.loc[x, 15]
                  }

        summation = 0
        for y in range(len(h_t_arr)):
            val = get_all_values(h_t_arr[y], dict_1)
            if val == 'no':
                summation = summation + (vote[y] * -1)
            elif val == 'yes':
                summation = summation + (vote[y] * 1)

        if summation >= 0:
            array.append('yes')
        else:
            array.append('no')

    return array


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
                       'weights': train.loc[:, 16],
                       'y': train.loc[:, 17]})

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

    h_t_arr = []
    vote = []
    h_t_err = []
    T_arr = []
    h_t_combined_err = []
    T = 501
    for x in range(1, T):
        ID3_hypothesis_of_level_2 = makeID3Tree(df, columns, labels, 1)
        ID3_arr = predict(ID3_hypothesis_of_level_2, train)
        error = find_weighted_classification_error(ID3_arr, df)
        print('Individual error at depth 2 for T = ', x, ' is:', error, '%')
        error = error / 100
        bracket_value_1 = (1 - error) / error
        alpha_vote = 0.5 * np.log(bracket_value_1)
        for y in range(5000):
            bracket_val = 0
            if ID3_arr[y] == df.iloc[y, 17]:
                bracket_val = -1 * (alpha_vote * 1)
            else:
                bracket_val = -1 * (alpha_vote * -1)
            df.iloc[y, 16] = df.iloc[y, 16] * math.exp(bracket_val)
        weight_sum = sum(df.iloc[:, 16])
        for y in range(5000):
            df.iloc[y, 16] = df.iloc[y, 16] / weight_sum

        h_t_arr.append(ID3_hypothesis_of_level_2)
        h_t_err.append(error)
        vote.append(alpha_vote)
        T_arr.append(x)

        combined_arr = combined_predict(h_t_arr, vote, train)
        error_1 = compute_error(combined_arr, df)
        h_t_combined_err.append(error_1)
        print('Combined error at depth 2 for T = ', x, ' is:', error_1 * 100, '%')

        print()

    print(h_t_arr)
    print(h_t_err)
    print(vote)
    print(T_arr)
    print(h_t_combined_err)

    plt.plot(T_arr, h_t_err)
    plt.xlabel('T value')
    plt.ylabel('Error at this T')
    plt.title('Individual error for Training')
    plt.show()

    plt.plot(T_arr, h_t_combined_err)
    plt.xlabel('T value')
    plt.ylabel('Error at this T')
    plt.title('Combined error for training')
    plt.show()

    return h_t_arr, vote, median_vals_for_test


def testing_part(h_t_arr, vote, median_vals):
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

    T = 501

    T_arr = []
    T_indiv_error = []
    T_combined_error = []

    for x in range(T - 1):
        T_arr.append(x+1)

        arr = predict(h_t_arr[x], test)
        error_1 = compute_error(arr, test)
        T_indiv_error.append(error_1)
        print('Individual error at depth 2 for T = ', x+1, ' is:', error_1*100, '%')

        indices = []
        for y in range(x + 1):
            indices.append(h_t_arr[y])
        combined_arr = combined_predict(indices, vote, test)
        error_2 = compute_error(combined_arr, test)
        T_combined_error.append(error_2)
        print('Combined error at depth 2 for T = ', x+1, ' is:', error_2 * 100, '%')
        print()

    plt.plot(T_arr, T_indiv_error)
    plt.xlabel('T value')
    plt.ylabel('Error at this T')
    plt.title('Individual error for Testing')
    plt.show()

    plt.plot(T_arr, T_combined_error)
    plt.xlabel('T value')
    plt.ylabel('Error at this T')
    plt.title('Combined error for Testing')
    plt.show()


def main():
    h_t_arr, vote, median_vals_for_test = training_part()
    testing_part(h_t_arr, vote, median_vals_for_test)


main()
