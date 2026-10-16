def matrix_shape(matrix):
    new_lst = []
    while isinstance(matrix, list):
        matrixes = len(matrix)
        new_lst.append(matrixes)
        matrix = matrix[0]
    return new_lst
