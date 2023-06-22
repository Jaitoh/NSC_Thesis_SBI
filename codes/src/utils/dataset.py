import numpy as np


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


if __name__ == "__main__":
    a, b, c = unravel_index(np.arange(20), (2, 3, 4))
    print(f"{a=}\n{b=}\n{c=}")
