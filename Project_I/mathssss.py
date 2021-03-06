import math
import numpy as np

def matrix_dim(A):

    dimensions = np.array([])

    try:
        dimensions.append(len(A[0]))
    except TypeError:
        dimensions.append(A[0])

    while True:
        counter = 0
        try:
            len(A[0])
        except TypeError:
            return dimensions
        dimensions.append(len(A))
        counter += 1
        A = A[0]

def dot_prod(v,w):

    if len(v) != len(w):
        raise ValueError("Hey! These vectors don't have the same length. len(v) , len(w)" , len(v) , len(w))

    dot_prod = 0

    for i in range(len(v)):
        dot_prod += v[i]*w[i]

    return dot_prod

def vecxmatrix(A,v):

    if len(A[0]) != len(v):
        raise ValueError("size of v does not match the rank of the matrix")
    
    result = np.array([0 for col in range(len(A))]).astype('float64')

    for i in range(len(A)):
        for j in range(len(v)):
            result[i] += A[i][j]*v[j]

    return result

def add_vector(v,w):

    if len(v) != len(w):
        raise ValueError("These vectors don't have the same length")

    return np.array([v[i]+w[i] for i in range(len(v))]).astype('float64')

def multiply_two_dim(A,B):

    if len(A[0]) != len(B):  
        raise ValueError("Cannot multiply these matricies")

    result = np.array( [ [ 0 for col in range(len(B[0])) ] for row in range(len(A)) ] ).astype('float64')

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(A[0])):

                result[i][j] += A[i][k]*B[k][j]

    return result

def sigmoid(x):
    return 1 / (1+math.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def transpose(matrix):

    A = matrix
    T = []

    for i in range(len(A[0])):
        new_vec = []
        for j in range(len(A)):
            new_vec.append(A[j][i])
        T.append(new_vec)

    return np.array(T)


def main():
    return 0

if __name__ == "__main__":
    main()