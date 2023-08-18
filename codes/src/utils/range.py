import numpy as np
from copy import deepcopy


# convert range of seqC
def x2seqC(x):
    """convert x into seqC
    x in range [0,1]
    seqC in range [-1,1]
    """

    # map the x from [0,1] to [-1, 1]
    x = x * 2 - 1
    # convert values to 1 decimal
    x = np.round(x, 1)
    # replace -1 with nan
    x[x == -1] = np.nan

    return x


def seqC2x(seqC):
    """convert seqC into x
    seqC in range [-1,1]
    x in range [0,1]
    """

    # map the seqC from [-1, 1] to [0,1]
    x = (seqC + 1) / 2
    # convert values to 1 decimal
    # x = np.round(x, 1)
    # replace nan with -1
    x[np.isnan(x)] = 0

    return x


def convert_samples_range(samples, original_limits, dest_limits):
    # map samples from original_limits to dest_limits
    # samples of shape (num_samples, num_params)
    samples = deepcopy(samples)

    for i in range(len(original_limits)):
        mapped_low = original_limits[i][0]
        mapped_up = original_limits[i][1]
        origin_low = dest_limits[i][0]
        origin_up = dest_limits[i][1]

        if len(samples.shape) == 1:
            samples[i] = (samples[i] - mapped_low) / (mapped_up - mapped_low) * (
                origin_up - origin_low
            ) + origin_low
        else:
            samples[..., i] = (samples[..., i] - mapped_low) / (mapped_up - mapped_low) * (
                origin_up - origin_low
            ) + origin_low

    return samples


def convert_array_range(f5, original_range, dest_range):
    """
    convert array from original_range [-1, 1] to dest_range [0, 1]
    """

    f5_dr = (f5 - original_range[0]) / (original_range[1] - original_range[0]) * (
        dest_range[1] - dest_range[0]
    ) + dest_range[0]

    return f5_dr
