import numpy as np


def color_from_seed(seed, order="rgb", num_type=int):
    np.random.seed(seed % (2**32))
    color_rgb = tuple((np.random.rand(3)))
    if order == "bgr":
        color = color_rgb[::-1]
    else:
        color = color_rgb
    if num_type is int:
        color = tuple((np.array(color) * 255).astype(int).tolist())
    return color
