import numpy as np
from typing import Union, Tuple


def expandtoarray(value: Union[float, np.ndarray], shape: Tuple[int, ...]) -> np.ndarray:
    """
    Expand a scalar value to an array of the given shape, validate or reshape arrays as needed.

    Args:
        value: A scalar, 1D array, or 2D array.
        shape: The target shape as a tuple.

    Returns:
        A NumPy array of the specified shape.

    Raises:
        ValueError: If the input array cannot be expanded to the target shape.
        TypeError: If the input is neither a scalar nor a NumPy array.
    """
    if np.isscalar(shape):
        shape = (shape, )

    if np.isscalar(value):
        return np.full(shape, value)

    if not isinstance(value, np.ndarray):
        raise TypeError(f"Expected scalar or numpy array, but got {type(value)}")

    if value.shape == shape:
        return value

    # Handle 1D array that matches the first dimension of the target shape
    if value.ndim == 1 and value.shape[0] == shape[0]:
        return np.tile(value[:, np.newaxis], (1, *shape[1:]))

    raise ValueError(
        f"Cannot broadcast input of shape {value.shape} to target shape {shape}"
    )
