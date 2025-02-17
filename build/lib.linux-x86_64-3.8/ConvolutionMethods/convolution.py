# convolution_methods.py
import numpy as np


def convolution1(A, B, N, delta, k):
    """
    Perform a custom convolution operation on arrays A and B.

    :param A: Input array A (numpy array).
    :param B: Input array B (numpy array).
    :param N: Integer parameter for the convolution.
    :param delta: Double parameter for the convolution.
    :param k: Integer parameter for the convolution.
    :return: Result of the convolution as a float.
    """
    if N > (len(A) - 1):
        return None

    # Perform the convolution operation
    result = np.sum(A[:N] * B[:N]) * delta * k
    return result

def convolution(A, B, delta, k):
    """
    Perform a convolution operation on arrays A and B.

    :param A: Input array A (numpy array).
    :param B: Input array B (numpy array).
    :param delta: Double parameter for the convolution.
    :param k: Integer parameter for the convolution.
    :return: Resulting array after convolution (numpy array).
    """
    M = len(A)
    out = np.zeros_like(A)

    # Perform the convolution operation
    for i in range(M):
        out[i] = np.sum(A[:i+1] * B[:i+1]) * delta * k

    return out

@staticmethod
def slice_test(A):
    """
    Copy the input array A to a new array.

    :param A: Input array A (numpy array).
    :return: A copy of the input array (numpy array).
    """
    return np.copy(A)