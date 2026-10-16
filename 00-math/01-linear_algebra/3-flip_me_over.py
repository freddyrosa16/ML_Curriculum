def matrix_transpose(matrix):
    result = []
    for column in range(len(matrix[0])):
        build = []
        for row in range(len(matrix)):
            build.append(matrix[row][column])
        result.append(build)
    return result