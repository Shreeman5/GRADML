import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


def compute_forward_pass(layer_0_nodes, layer_01_matrix, layer_12_matrix, layer_23_matrix, width, base_value,
                         top_value):
    layer_1_nodes = []
    for c in range(width - 1):
        sum_val = 0
        for r in range(base_value):
            sum_val += layer_0_nodes[r] * layer_01_matrix[r][c]
        sigma_value = 1 / (1 + math.exp(-sum_val))
        layer_1_nodes.append(sigma_value)
    layer_1_nodes.append(1)
    # print('Layer1 nodes:', layer_1_nodes)

    layer_2_nodes = []
    for c in range(width - 1):
        sum_val = 0
        for r in range(width):
            sum_val += layer_1_nodes[r] * layer_12_matrix[r][c]
        sigma_value = 1 / (1 + math.exp(-sum_val))
        layer_2_nodes.append(sigma_value)
    layer_2_nodes.append(1)
    # print('Layer2 nodes:', layer_2_nodes)

    y_val = 0
    for c in range(top_value):
        for r in range(width):
            y_val += layer_2_nodes[r] * layer_23_matrix[r][c]

    return y_val, layer_1_nodes, layer_2_nodes


def back_propagation_wrt_layer_23_weights(layer_23_matrix, layer_2_nodes, y_i, y_star, learning_rate):
    for c in range(layer_23_matrix.shape[1]):
        for r in range(layer_23_matrix.shape[0]):
            derivative = (y_i - y_star) * layer_2_nodes[r]
            layer_23_matrix[r][c] = layer_23_matrix[r][c] - (learning_rate * derivative)
    return layer_23_matrix


def back_propagation_wrt_layer_12_weights(layer_23_matrix, layer_12_matrix, layer_2_nodes, layer_1_nodes, y_i, y_star,
                                          learning_rate):
    dldy = y_i - y_star
    row = 0
    for c in range(layer_12_matrix.shape[1]):
        common_term = layer_23_matrix[row][0] * layer_2_nodes[c] * (1 - layer_2_nodes[c])
        for r in range(layer_12_matrix.shape[0]):
            differing_term = layer_1_nodes[r]
            layer_12_matrix[r][c] = layer_12_matrix[r][c] - (learning_rate * dldy * common_term * differing_term)
        row = row + 1
    return layer_12_matrix


def back_propagation_wrt_layer_01_weights(layer_23_matrix, layer_12_matrix, layer_01_matrix, layer_2_nodes,
                                          layer_1_nodes, layer_0_nodes, y_i, y_star, learning_rate):
    # print(layer_12_matrix)
    dldy = y_i - y_star
    common_term = []
    for x in range(layer_23_matrix.shape[0] - 1):
        common_term.append(layer_23_matrix[x][0] * dldy * layer_2_nodes[x] * (1 - layer_2_nodes[x]))
    # print('C:', common_term)

    row = 0
    for c in range(layer_01_matrix.shape[1]):
        layer_12_row = layer_12_matrix[c]
        next_common_term = []
        for x in range(len(layer_12_row)):
            next_common_term.append(layer_12_row[x] * common_term[x])
        # print('LC:', next_common_term)
        for r in range(layer_01_matrix.shape[0]):
            final_term = layer_0_nodes[r] * layer_1_nodes[row] * (1 - layer_1_nodes[row])
            # print('Final_term:', final_term)
            next_next_common_term = []
            for y in range(len(next_common_term)):
                next_next_common_term.append(next_common_term[y] * final_term)
            # print('NNT: ', next_next_common_term)
            derivative = sum(next_next_common_term)
            # print('Derivative: ', derivative)
            layer_01_matrix[r][c] = layer_01_matrix[r][c] - (learning_rate * derivative)
        row = row + 1
    # print('Here:', layer_01_matrix)
    # print(layer_01_matrix.shape)
    return layer_01_matrix


def training(train, layer_01_matrix, layer_12_matrix, layer_23_matrix, width, base_value, top_value):
    learning_rate = 0.05
    d_val = 1
    T = 10
    loss = []

    for x in range(T):
        df = train.sample(frac=1).reset_index(drop=True)
        # df = train
        learning_rate = learning_rate / (1 + ((learning_rate * x) / d_val))
        for y in range(df.shape[0]):
            layer_0_nodes = df.iloc[y, 0:base_value]
            # print(layer_0_nodes)
            y_i, layer_1_nodes, layer_2_nodes = compute_forward_pass(layer_0_nodes, layer_01_matrix, layer_12_matrix,
                                                                     layer_23_matrix, width,
                                                                     base_value, top_value)
            y_star = df.iloc[y, base_value]
            # print('y_i:', y_i)
            # print('y_star_i: ', y_star)
            inside_bracket_value = y_i - y_star
            loss_function = 0.5 * (inside_bracket_value ** 2)
            loss.append(loss_function)
            # # print('Loss:', loss_function)

            layer_01_matrix = back_propagation_wrt_layer_01_weights(layer_23_matrix, layer_12_matrix, layer_01_matrix,
                                                                    layer_2_nodes, layer_1_nodes, layer_0_nodes, y_i,
                                                                    y_star, learning_rate)
            # layer_01_matrix_holder.append(layer_01_matrix)
            # print('Layer01 matrix:', layer_01_matrix)

            layer_12_matrix = back_propagation_wrt_layer_12_weights(layer_23_matrix, layer_12_matrix, layer_2_nodes,
                                                                    layer_1_nodes, y_i,
                                                                    y_star, learning_rate)
            # print('Layer12 matrix:', layer_12_matrix)

            layer_23_matrix = back_propagation_wrt_layer_23_weights(layer_23_matrix, layer_2_nodes, y_i, y_star,
                                                                    learning_rate)
            # print('layer23 matrix:', layer_23_matrix)

    num_array = []
    for z in range(len(loss)):
        num_array.append(z)

    plt.plot(num_array, loss)
    plt.xlabel('training example i')
    plt.ylabel('loss at this i')
    plt.title('over time')
    plt.show()

    # for row in range(layer_01_matrix.shape[0]):
    #     for column in range(layer_01_matrix.shape[1]):
    #         values = []
    #         for n in range(len(layer_01_matrix_holder)):
    #             matrix = layer_01_matrix_holder[n]
    #             # print(matrix)
    #             values.append(matrix[row][column])
    #         plt.plot(num_array, values)
    #         plt.xlabel('number')
    #         plt.ylabel('matrixrc_value')
    #         plt.title('over time')
    #         plt.show()

    return layer_01_matrix, layer_12_matrix, layer_23_matrix


def initialize_matrices():
    width = 100
    base_value = 5
    top_value = 1

    layer_01_matrix = np.random.randn(base_value, width - 1)
    layer_01_matrix[base_value - 1] = 0
    # print('Layer01 weights:', layer_01_matrix)
    # print('Shape:', layer_01_matrix.shape)

    layer_12_matrix = np.random.randn(width, width - 1)
    layer_12_matrix[width - 1] = 0
    # print('Layer12 weights:', layer_12_matrix)
    # print('Shape:', layer_12_matrix.shape)

    layer_23_matrix = np.random.randn(width, top_value)
    layer_23_matrix[width - 1] = 0
    # print('Layer23 weights:', layer_23_matrix)
    # print('Shape:', layer_23_matrix.shape)

    return layer_01_matrix, layer_12_matrix, layer_23_matrix, width, base_value, top_value


def find_y_val_using_forward_pass(layer_01_matrix, layer_12_matrix, layer_23_matrix,
                                  width, base_value, top_value, train):
    counter = 0
    for y in range(train.shape[0]):
        layer_0_nodes = train.iloc[y, 0:base_value]

        layer_1_nodes = []
        for c in range(width - 1):
            sum_val = 0
            for r in range(base_value):
                sum_val += layer_0_nodes[r] * layer_01_matrix[r][c]
                # print('SV:', sum_val)
            sigma_value = 1 / (1 + math.exp(-sum_val))
            layer_1_nodes.append(sigma_value)
        layer_1_nodes.append(1)
        # print('Layer1 nodes:', layer_1_nodes)

        layer_2_nodes = []
        for c in range(width - 1):
            sum_val = 0
            for r in range(width):
                sum_val += layer_1_nodes[r] * layer_12_matrix[r][c]
            sigma_value = 1 / (1 + math.exp(-sum_val))
            layer_2_nodes.append(sigma_value)
        layer_2_nodes.append(1)
        # print('Layer2 nodes:', layer_2_nodes)

        y_val = 0
        for c in range(top_value):
            for r in range(width):
                y_val += layer_2_nodes[r] * layer_23_matrix[r][c]

        if y_val <= 0.5:
            y_val = 0
        else:
            y_val = 1

        if y_val != train.iloc[y, 5]:
            counter = counter + 1
        # print('Y: ', y_val, 'Y_star: ', train.iloc[y, 5])
    print('Counter: ', counter)
    return (counter / train.shape[0]) * 100


def main():
    layer_01_matrix, layer_12_matrix, layer_23_matrix, width, base_value, top_value = initialize_matrices()
    train = pd.read_csv('train.csv', header=None)
    layer_01_matrix, layer_12_matrix, layer_23_matrix = training(train, layer_01_matrix,
                                                                 layer_12_matrix,
                                                                 layer_23_matrix, width,
                                                                 base_value, top_value)

    training_error = find_y_val_using_forward_pass(layer_01_matrix, layer_12_matrix,
                                                   layer_23_matrix, width, base_value, top_value, train)
    print('TR:', training_error)
    test = pd.read_csv('test.csv', header=None)
    testing_error = find_y_val_using_forward_pass(layer_01_matrix, layer_12_matrix,
                                                  layer_23_matrix, width, base_value, top_value, test)
    print('TE:', testing_error)


main()
