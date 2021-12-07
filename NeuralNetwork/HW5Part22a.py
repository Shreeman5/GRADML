import math


def compute_forward_pass(layer_0_nodes, layer_01_matrix, layer_12_matrix, layer_23_matrix, width, base_value,
                         top_value):
    layer_1_nodes = []
    for c in range(width - 1):
        sum_val = 0
        for r in range(base_value):
            sum_val += layer_0_nodes[r] * layer_01_matrix[r][c]
            # print('SV:', sum_val)
        sigma_value = 1 / (1 + math.exp(-sum_val))
        # print('Sigma1:', sigma_value)
        layer_1_nodes.append(sigma_value)
    layer_1_nodes.append(1)

    layer_2_nodes = []
    for c in range(width - 1):
        sum_val = 0
        for r in range(width):
            sum_val += layer_1_nodes[r] * layer_12_matrix[r][c]
            # print('SV1:', sum_val)
        sigma_value = 1 / (1 + math.exp(-sum_val))
        # print('Sigma2:', sigma_value)
        layer_2_nodes.append(sigma_value)
    layer_2_nodes.append(1)

    y_val = 0
    for c in range(top_value):
        for r in range(width):
            y_val += layer_2_nodes[r] * layer_23_matrix[r][c]

    return y_val, layer_1_nodes, layer_2_nodes


def back_propagation_wrt_layer_23_weights(layer_23_matrix, layer_2_nodes, y, y_star):
    for c in range(1):
        for r in range(3):
            derivative = (y - y_star) * layer_2_nodes[r]
            layer_23_matrix[r][c] = derivative
    return layer_23_matrix


def back_propagation_wrt_layer_12_weights(layer_23_matrix, layer_12_matrix, layer_2_nodes, layer_1_nodes, y, y_star):
    # print(layer_12_matrix)
    # print(layer_23_matrix)
    dldy = y - y_star
    row = 0
    for c in range(2):
        common_term = layer_23_matrix[row][0] * layer_2_nodes[c] * (1 - layer_2_nodes[c])
        for r in range(3):
            differing_term = layer_1_nodes[r]
            layer_12_matrix[r][c] = dldy * common_term * differing_term
        row = row + 1
    return layer_12_matrix


def back_propagation_wrt_layer_01_weights(layer_23_matrix, layer_12_matrix, layer_01_matrix,
                                          layer_2_nodes, layer_1_nodes, layer_0_nodes, y, y_star):
    # print(layer_12_matrix)
    dldy = y - y_star
    common_term = []
    for x in range(len(layer_23_matrix) - 1):
        common_term.append(dldy * layer_23_matrix[x][0] * layer_2_nodes[x] * (1 - layer_2_nodes[x]))

    row = 0
    for c in range(2):
        layer_12_row = layer_12_matrix[c]
        next_common_term = []
        for x in range(len(layer_12_row)):
            next_common_term.append(layer_12_row[x] * common_term[x])
        for r in range(3):
            final_term = layer_0_nodes[r] * layer_1_nodes[row] * (1 - layer_1_nodes[row])
            # print('Final_term:', final_term)
            next_next_common_term = []
            for y in range(len(next_common_term)):
                next_next_common_term.append(next_common_term[y] * final_term)
            # print('NNT: ', next_next_common_term)
            derivative = sum(next_next_common_term)
            # print('Derivative: ', derivative)
            layer_01_matrix[r][c] = derivative
        row = row + 1
    # print(layer_01_matrix.shape)
    return layer_01_matrix


def training(layer_01_matrix, layer_12_matrix, layer_23_matrix, width, base_value, top_value):
    layer_0_nodes = [1, 1, 1]

    # print(layer_0_nodes)
    y, layer_1_nodes, layer_2_nodes = compute_forward_pass(layer_0_nodes, layer_01_matrix, layer_12_matrix,
                                                           layer_23_matrix, width,
                                                           base_value, top_value)
    y_star = 1

    layer_01_matrix = back_propagation_wrt_layer_01_weights(layer_23_matrix, layer_12_matrix, layer_01_matrix,
                                                            layer_2_nodes, layer_1_nodes, layer_0_nodes, y, y_star)

    print('Layer01 matrix:', layer_01_matrix)

    layer_12_matrix = back_propagation_wrt_layer_12_weights(layer_23_matrix, layer_12_matrix, layer_2_nodes,
                                                            layer_1_nodes, y, y_star)

    print('Layer12 matrix:', layer_12_matrix)

    layer_23_matrix = back_propagation_wrt_layer_23_weights(layer_23_matrix, layer_2_nodes, y, y_star)

    print('layer23 matrix:', layer_23_matrix)


def initialize_matrices():
    width = 3
    base_value = 3
    top_value = 1

    layer_01_matrix = [[0 for j in range(width - 1)] for i in range(base_value)]
    layer_01_matrix[0][0] = -2
    layer_01_matrix[0][1] = 2
    layer_01_matrix[1][0] = -3
    layer_01_matrix[1][1] = 3
    layer_01_matrix[2][0] = -1
    layer_01_matrix[2][1] = 1
    # print('Layer01 weights:', layer_01_matrix)

    layer_12_matrix = [[0 for j in range(width - 1)] for i in range(width)]
    layer_12_matrix[0][0] = -2
    layer_12_matrix[0][1] = 2
    layer_12_matrix[1][0] = -3
    layer_12_matrix[1][1] = 3
    layer_12_matrix[2][0] = -1
    layer_12_matrix[2][1] = 1
    # print('Layer12 weights:', layer_12_matrix)

    layer_23_matrix = [[0 for j in range(top_value)] for i in range(width)]
    layer_23_matrix[0][0] = 2
    layer_23_matrix[1][0] = -1.5
    layer_23_matrix[2][0] = -1
    # print('Layer23 weights:', layer_23_matrix)

    return layer_01_matrix, layer_12_matrix, layer_23_matrix, width, base_value, top_value


def main():
    layer_01_matrix, layer_12_matrix, layer_23_matrix, width, base_value, top_value = initialize_matrices()
    training(layer_01_matrix, layer_12_matrix, layer_23_matrix, width, base_value, top_value)


main()
