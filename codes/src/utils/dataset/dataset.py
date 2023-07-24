import numpy as np
import torch


def pad_seqC_with_nans_to_len15(seqC):
    """
    pad seq with nans to length 15, along the last dimension
    """

    seqC_shape = seqC.shape
    dur = seqC_shape[-1]

    padding_shape = [*seqC_shape[:-1], 15 - dur]
    padding_array = np.full((padding_shape), np.nan)

    seqC = np.concatenate([seqC, padding_array], axis=-1)

    return seqC


def update_prior_min_max(prior_min, prior_max, ignore_ss, normalize):
    if not ignore_ss:
        unnormed_prior_min = prior_min
        unnormed_prior_max = prior_max
    else:
        unnormed_prior_min = prior_min[:1] + prior_min[3:]
        unnormed_prior_max = prior_max[:1] + prior_max[3:]

    if normalize:
        prior_min = np.zeros_like(unnormed_prior_min)
        prior_max = np.ones_like(unnormed_prior_max)
    else:
        prior_min = unnormed_prior_min
        prior_max = unnormed_prior_max

    return prior_min, prior_max, unnormed_prior_min, unnormed_prior_max


def choose_theta(
    num_chosen_theta_each_set,
    max_theta_in_a_set,
    theta_chosen_mode="random",
):
    """choose theta and return theta idx from the set of theta in the dataset"""

    assert (
        num_chosen_theta_each_set <= max_theta_in_a_set
    ), f"num_chosen_theta_each_set={num_chosen_theta_each_set} > max_theta_in_a_set={max_theta_in_a_set}"

    # choose randomly num_chosen_theta_each_set from all the theta in the set
    if theta_chosen_mode == "random":
        theta_idx = np.random.choice(
            max_theta_in_a_set, num_chosen_theta_each_set, replace=False
        )
        return np.sort(theta_idx), len(theta_idx)

    # choose first 80% as training set from num_chosen_theta_each_set
    elif theta_chosen_mode.startswith("first"):  # first_80
        percentage = eval(theta_chosen_mode[-2:]) / 100
        theta_idx = np.arange(int(num_chosen_theta_each_set * percentage))
        return theta_idx, len(theta_idx)

    # choose last 20% as validation set from num_chosen_theta_each_set
    elif theta_chosen_mode.startswith("last"):  # last_20
        percentage = 1 - eval(theta_chosen_mode[-2:]) / 100
        theta_idx = np.arange(
            int(num_chosen_theta_each_set * percentage), num_chosen_theta_each_set
        )
        return theta_idx, len(theta_idx)

    else:
        raise ValueError(f"Invalid theta_chosen_mode: {theta_chosen_mode}")


def get_len_seqC(seqC_process, summary_type):
    """
    The function `get_len_seqC` returns the length of a sequence based on the given `seqC_process` and
    `summary_type` parameters.
    """

    if seqC_process == "norm":
        # seqC_shape = f[chosen_set_names[0]]['seqC_normed'].shape[1]
        L = 15
    elif seqC_process == "summary":
        if summary_type == 0:
            # seqC_shape = f[chosen_set_names[0]]['seqC_summary_0'].shape[1]
            L = 11
        elif summary_type == 1:
            # seqC_shape = f[chosen_set_names[0]]['seqC_summary_1'].shape[1]
            L = 8
    return L


def generate_permutations(N, K):
    """
    Generate random permutations.

    Args:
        N (int): The number of permutations to generate.
        K (int): The length of each permutation.

    Returns:
        torch.Tensor: A tensor of shape (N, K) containing random permutations.

    """
    return torch.rand(N, K).argsort(dim=-1)


def unravel_index(index, shape):
    """
    Translates a flat index into a multi-dimensional index in a multi-dimensional array with the specified shape.

    The function works in a similar way to numpy's unravel_index, but only works for 1-dimensional indices.

    Args:
        index (int): The flat index into the multi-dimensional array. This index is assumed to be in row-major (C-style) order.
        shape (tuple): The shape of the multi-dimensional array. Each element of the tuple represents a dimension size.

    Returns:
        tuple: The multi-dimensional index corresponding to the flat index in the array of the given shape.

    Raises:
        ValueError: If 'index' is greater than the total elements in 'shape'.

    Example:
        >>> unravel_index(22, (5, 5))
        (4, 2)

        This means that the element at the 22nd position in the flattened version of a 5x5 matrix is located at the 4th row and 2nd column in the unflattened matrix. Please remember that indices start from 0.

    """
    total_elements = 1
    for dim in shape:
        total_elements *= dim
    if index >= total_elements:
        raise ValueError(
            "Index out of bound. It should be less than total elements in shape."
        )

    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def process_theta(
    theta,
    ignore_ss=False,
    normalize_theta=False,
    unnormed_prior_min=None,
    unnormed_prior_max=None,
):
    """
    theta: (n_sets, n_T, n_theta)
    ignore_ss: whether to ignore the second and third parameters of theta
    normalize_theta: whether to normalize theta
    """

    if normalize_theta:
        for i in range(theta.shape[-1]):
            theta[:, :, i] = (theta[:, :, i] - unnormed_prior_min[i]) / (
                unnormed_prior_max[i] - unnormed_prior_min[i]
            )

    if ignore_ss:
        theta = torch.cat((theta[:, :, :1], theta[:, :, 3:]), dim=-1)

    return theta


def process_theta_2D(
    theta,
    ignore_ss=False,
    normalize_theta=False,
    unnormed_prior_min=None,
    unnormed_prior_max=None,
):
    """
    theta: (n_T, n_theta)
    ignore_ss: whether to ignore the second and third parameters of theta
    normalize_theta: whether to normalize theta
    """

    if normalize_theta:
        for i in range(theta.shape[-1]):
            theta[:, i] = (theta[:, i] - unnormed_prior_min[i]) / (
                unnormed_prior_max[i] - unnormed_prior_min[i]
            )

    if ignore_ss:
        theta = torch.cat((theta[:, :1], theta[:, 3:]), dim=-1)

    return theta


def apply_advanced_indexing_along_dim1(tensor, indices):
    idx0 = torch.arange(tensor.size(0))[:, None].expand(
        indices.size(0), indices.size(1)
    )
    return tensor[idx0, indices]


if __name__ == "__main__":
    a, b, c = unravel_index(np.arange(20), (2, 3, 4))
    print(f"{a=}\n{b=}\n{c=}")
