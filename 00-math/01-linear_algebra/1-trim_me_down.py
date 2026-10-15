matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = []
for i in matrix:
    the_middle.append(i[2:4])


print(f"The middle columns of the matrix are: {the_middle}")