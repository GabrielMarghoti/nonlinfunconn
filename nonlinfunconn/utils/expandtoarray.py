import numpy as np
from typing import Union, Tuple


def expandtoarray(value: Union[float, np.ndarray], shape: Tuple[int, ...]) -> np.ndarray:
        """
        Expand a scalar value to an array of the given shape, expand 1D arrays to 2D if necessary,
        or validate an existing array.

        Args:
            value: A scalar, 1D array, or 2D array.
            shape: The target shape as a tuple (e.g., (rows, cols)).

        Returns:
            A NumPy array of the specified shape.

        Raises:
            ValueError: If the input array cannot be expanded to the target shape.
            TypeError: If the input is neither a scalar nor a NumPy array.
        """
        if np.isscalar(value):
            return np.full(shape, value)

        if not isinstance(value, np.ndarray):
            raise TypeError(f"Expected scalar or numpy array, but got {type(value)}")

        if value.shape == shape:
            return value

        if value.ndim == 1:
            out = np.zeros(shape)
            for i in range(shape[0]):
                out[i, :] = value
            return out
        else:
            raise ValueError(
                f"1D array length {value.shape[0]} does not match either dimension of the target shape {shape}"
            )
        