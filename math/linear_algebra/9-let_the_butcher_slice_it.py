#!/usr/bin/env python3
import numpy as np

def main():
    """
    This function creates a 4x6 matrix, slices it into three specific parts,
    and prints the results.
    """
    matrix = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],
                       [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]])

    # Slice the matrix to get the middle two rows
    mat1 = matrix[1:3, :]

    # Slice the matrix to get the middle two columns
    mat2 = matrix[:, 2:4]

    # Slice the matrix to get the bottom-right, square, 3x3 matrix
    mat3 = matrix[1:, 3:]

    # Print the results
    print("The middle two rows of the matrix are:\n{}".format(mat1))
    print("The middle two columns of the matrix are:\n{}".format(mat2))
    print("The bottom-right, square, 3x3 matrix is:\n{}".format(mat3))

if __name__ == "__main__":
    main()
