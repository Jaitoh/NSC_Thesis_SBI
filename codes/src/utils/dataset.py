import numpy as np
import torch


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
    if ignore_ss:
        theta = torch.cat((theta[:, :, :1], theta[:, :, 3:]), dim=-1)

    if normalize_theta:
        for i in range(theta.shape[-1]):
            theta[:, :, i] = (theta[:, :, i] - unnormed_prior_min[i]) / (
                unnormed_prior_max[i] - unnormed_prior_min[i]
            )
    return theta


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


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


if __name__ == "__main__":
    a, b, c = unravel_index(np.arange(20), (2, 3, 4))
    print(f"{a=}\n{b=}\n{c=}")
