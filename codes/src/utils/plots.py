import cv2
import matplotlib.pyplot as plt
from sbi.analysis.plot import _update, _format_axis, prepare_for_plot, probs2contours, ensure_numpy
from scipy.stats import binom, gaussian_kde
import numpy as np
import matplotlib as mpl

from pathlib import Path
import sys

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")
from utils.inference import (
    get_posterior,
    load_stored_config,
    sampling_from_posterior,
)
from utils.range import convert_samples_range


def load_img(
    img_path,
    ax,
    title,
    crop=False,
    x_start=None,
    x_end=None,
    y_start=None,
    y_end=None,
    print_shape=False,
):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # print total number of pixels
    if print_shape:
        print(f"Image number of pixels: {img.shape[0], img.shape[1]}")

    # Crop image
    if crop:
        img = img[y_start:y_end, x_start:x_end]

    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def _arrange_plots(diag_func, upper_func, dim, limits, points, opts, fig=None, axes=None):
    """
    Arranges the plots for any function that plots parameters either in a row of 1D
    marginals or a pairplot setting.

    Args:
        diag_func: Plotting function that will be executed for the diagonal elements of
            the plot (or the columns of a row of 1D marginals). It will be passed the
            current `row` (i.e. which parameter that is to be plotted) and the `limits`
            for all dimensions.
        upper_func: Plotting function that will be executed for the upper-diagonal
            elements of the plot. It will be passed the current `row` and `col` (i.e.
            which parameters are to be plotted and the `limits` for all dimensions. None
            if we are in a 1D setting.
        dim: The dimensionality of the density.
        limits: Limits for each parameter.
        points: Additional points to be scatter-plotted.
        opts: Dictionary built by the functions that call `_arrange_plots`. Must
            contain at least `labels`, `subset`, `figsize`, `subplots`,
            `fig_subplots_adjust`, `title`, `title_format`, ..
        fig: matplotlib figure to plot on.
        axes: matplotlib axes corresponding to fig.

    Returns: figure and axis
    """

    # Prepare points
    if points is None:
        points = []
    if type(points) != list:
        points = ensure_numpy(points)  # type: ignore
        points = [points]
    points = [np.atleast_2d(p) for p in points]
    points = [np.atleast_2d(ensure_numpy(p)) for p in points]

    # Prepare labels
    if opts["labels"] == [] or opts["labels"] is None:
        labels_dim = ["dim {}".format(i + 1) for i in range(dim)]
    else:
        labels_dim = opts["labels"]

    # Prepare ticks
    if opts["ticks"] == [] or opts["ticks"] is None:
        ticks = None
    else:
        if len(opts["ticks"]) == 1:
            ticks = [opts["ticks"][0] for _ in range(dim)]
        else:
            ticks = opts["ticks"]

    # Figure out if we subset the plot
    subset = opts["subset"]
    if subset is None:
        rows = cols = dim
        subset = [i for i in range(dim)]
    else:
        if type(subset) == int:
            subset = [subset]
        elif type(subset) == list:
            pass
        else:
            raise NotImplementedError
        rows = cols = len(subset)
    flat = upper_func is None
    if flat:
        rows = 1
        opts["lower"] = None

    # Create fig and axes if they were not passed.
    if fig is None or axes is None:
        fig, axes = plt.subplots(rows, cols, figsize=opts["figsize"], **opts["subplots"])
    else:
        assert axes.shape == (
            rows,
            cols,
        ), f"Passed axes must match subplot shape: {rows, cols}."
    # Cast to ndarray in case of 1D subplots.
    axes = np.array(axes).reshape(rows, cols)

    # Style figure
    fig.subplots_adjust(**opts["fig_subplots_adjust"])
    fig.suptitle(opts["title"], **opts["title_format"])

    # Style axes
    row_idx = -1
    for row in range(dim):
        if row not in subset:
            continue

        if not flat:
            row_idx += 1

        col_idx = -1
        for col in range(dim):
            if col not in subset:
                continue
            else:
                col_idx += 1

            if flat:
                current = "diag"
            elif row == col:
                current = "diag"
            elif row < col:
                current = "upper"
            else:
                current = "lower"

            ax = axes[row_idx, col_idx]
            plt.sca(ax)

            # Background color
            if current in opts["fig_bg_colors"] and opts["fig_bg_colors"][current] is not None:
                ax.set_facecolor(opts["fig_bg_colors"][current])

            # Axes
            if opts[current] is None:
                ax.axis("off")
                continue

            # Limits
            ax.set_xlim((limits[col][0], limits[col][1]))
            if current != "diag":
                ax.set_ylim((limits[row][0], limits[row][1]))

            # Ticks
            if ticks is not None:
                ax.set_xticks((ticks[col][0], ticks[col][1]))
                if current != "diag":
                    ax.set_yticks((ticks[row][0], ticks[row][1]))

            # Despine
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_position(("outward", opts["despine"]["offset"]))

            # Formatting axes
            if current == "diag":
                if opts["lower"] is None or col == dim - 1 or flat:
                    _format_axis(
                        ax,
                        xhide=False,
                        xlabel=labels_dim[col],
                        yhide=False,
                        ylabel="density",
                        tickformatter=opts["tickformatter"],
                    )
                else:
                    _format_axis(ax, xhide=True, yhide=True)
            else:  # off-diagnoals
                if row == dim - 1:
                    _format_axis(
                        ax,
                        xhide=False,
                        xlabel=labels_dim[col],
                        yhide=True,
                        tickformatter=opts["tickformatter"],
                    )
                else:
                    _format_axis(ax, xhide=True, yhide=True)
            if opts["tick_labels"] is not None:
                ax.set_xticklabels(
                    (
                        str(opts["tick_labels"][col][0]),
                        str(opts["tick_labels"][col][1]),
                    )
                )

            # Diagonals
            if current == "diag":
                diag_func(row=col, limits=limits)

                if len(points) > 0:
                    extent = ax.get_ylim()
                    for n, v in enumerate(points):
                        plt.plot(
                            [v[:, col], v[:, col]],
                            extent,
                            color=opts["points_colors"][n],
                            **opts["points_diag"],
                        )

            # Off-diagonals
            else:
                upper_func(
                    row=row,
                    col=col,
                    limits=limits,
                )

                if len(points) > 0:
                    for n, v in enumerate(points):
                        plt.plot(
                            v[:, col],
                            v[:, row],
                            color=opts["points_colors"][n],
                            **opts["points_offdiag"],
                        )

    if len(subset) < dim:
        if flat:
            ax = axes[0, len(subset) - 1]
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            text_kwargs = {"fontsize": plt.rcParams["font.size"] * 2.0}
            ax.text(x1 + (x1 - x0) / 8.0, (y0 + y1) / 2.0, "...", **text_kwargs)
        else:
            for row in range(len(subset)):
                ax = axes[row, len(subset) - 1]
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                text_kwargs = {"fontsize": plt.rcParams["font.size"] * 2.0}
                ax.text(x1 + (x1 - x0) / 8.0, (y0 + y1) / 2.0, "...", **text_kwargs)
                if row == len(subset) - 1:
                    ax.text(
                        x1 + (x1 - x0) / 12.0,
                        y0 - (y1 - y0) / 1.5,
                        "...",
                        rotation=-45,
                        **text_kwargs,
                    )

    return fig, axes


def _get_default_opts():
    """Return default values for plotting specs."""

    return {
        # 'lower': None,     # hist/scatter/None  # TODO: implement
        # title and legend
        "title": None,
        "legend": False,
        # labels
        "labels_points": [],  # for points
        "labels_samples": [],  # for samples
        # colors
        "samples_colors": ["k", "r"] + plt.rcParams["axes.prop_cycle"].by_key()["color"],
        # ticks
        "tickformatter": mpl.ticker.FormatStrFormatter("%g"),
        "tick_labels": None,
        # options for hist
        "hist_diag": {"alpha": 1.0, "bins": 50, "density": False, "histtype": "step"},
        "hist_offdiag": {
            # 'edgecolor': 'none',
            # 'linewidth': 0.0,
            "bins": 50,
        },
        # options for kde
        "kde_diag": {"bw_method": "scott", "bins": 50, "color": "black"},
        "kde_offdiag": {"bw_method": "scott", "bins": 50},
        # options for contour
        "contour_offdiag": {"levels": [0.68], "percentile": True},
        # options for scatter
        "scatter_offdiag": {
            "alpha": 0.5,
            "edgecolor": "none",
            "rasterized": False,
        },
        "scatter_diag": {},
        # options for plot
        "plot_offdiag": {},
        # formatting points (scale, markers)
        "points_diag": {},
        "points_offdiag": {
            "marker": ".",
            "markersize": 20,
        },
        # other options
        "fig_bg_colors": {"upper": None, "diag": None, "lower": None},
        "fig_subplots_adjust": {
            "top": 0.9,
        },
        "subplots": {},
        "despine": {
            "offset": 5,
        },
        "title_format": {"fontsize": 12},
    }


def get_diag_func(samples, limits, opts, **kwargs):
    """
    Returns the diag_func which returns the 1D marginal plot for the parameter
    indexed by row.
    """

    def diag_func(row, **kwargs):
        if len(samples) > 0:
            for n, v in enumerate(samples):
                if opts["diag"][n] == "hist":
                    plt.hist(v[:, row], color=opts["samples_colors"][n], **opts["hist_diag"])
                elif opts["diag"][n] == "kde":
                    density = gaussian_kde(v[:, row], bw_method=opts["kde_diag"]["bw_method"])
                    xs = np.linspace(limits[row, 0], limits[row, 1], opts["kde_diag"]["bins"])
                    ys = density(xs)
                    plt.plot(
                        xs,
                        ys,
                        color=opts["samples_colors"][n],
                    )
                    # show y axis on the left
                    plt.ylabel("density")

                elif "upper" in opts.keys() and opts["upper"][n] == "scatter":
                    for single_sample in v:
                        plt.axvline(
                            single_sample[row],
                            color=opts["samples_colors"][n],
                            **opts["scatter_diag"],
                        )
                else:
                    pass

    return diag_func


def pairplot(
    samples,
    points=None,
    limits=None,
    subset=None,
    upper="hist",
    diag="hist",
    figsize=(10, 10),
    labels=None,
    ticks=[],
    points_colors=plt.rcParams["axes.prop_cycle"].by_key()["color"],
    fig=None,
    axes=None,
    **kwargs,
):
    opts = _get_default_opts()
    # update the defaults dictionary by the current values of the variables (passed by
    # the user)

    opts = _update(opts, locals())
    opts = _update(opts, kwargs)

    samples, dim, limits = prepare_for_plot(samples, limits)

    # Prepare diag/upper/lower
    if type(opts["diag"]) is not list:
        opts["diag"] = [opts["diag"] for _ in range(len(samples))]
    if type(opts["upper"]) is not list:
        opts["upper"] = [opts["upper"] for _ in range(len(samples))]
    # if type(opts['lower']) is not list:
    #    opts['lower'] = [opts['lower'] for _ in range(len(samples))]
    opts["lower"] = None

    diag_func = get_diag_func(samples, limits, opts, **kwargs)

    def upper_func(row, col, limits, **kwargs):
        if len(samples) > 0:
            for n, v in enumerate(samples):
                if opts["upper"][n] == "hist" or opts["upper"][n] == "hist2d":
                    hist, xedges, yedges = np.histogram2d(
                        v[:, col],
                        v[:, row],
                        range=[
                            [limits[col][0], limits[col][1]],
                            [limits[row][0], limits[row][1]],
                        ],
                        **opts["hist_offdiag"],
                    )
                    plt.imshow(
                        hist.T,
                        origin="lower",
                        extent=(
                            xedges[0],
                            xedges[-1],
                            yedges[0],
                            yedges[-1],
                        ),
                        aspect="auto",
                    )

                elif opts["upper"][n] in [
                    "kde",
                    "kde2d",
                    "contour",
                    "contourf",
                ]:
                    density = gaussian_kde(
                        v[:, [col, row]].T,
                        bw_method=opts["kde_offdiag"]["bw_method"],
                    )
                    X, Y = np.meshgrid(
                        np.linspace(
                            limits[col][0],
                            limits[col][1],
                            opts["kde_offdiag"]["bins"],
                        ),
                        np.linspace(
                            limits[row][0],
                            limits[row][1],
                            opts["kde_offdiag"]["bins"],
                        ),
                    )
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    Z = np.reshape(density(positions).T, X.shape)

                    if opts["upper"][n] == "kde" or opts["upper"][n] == "kde2d":
                        plt.imshow(
                            Z,
                            extent=(
                                limits[col][0],
                                limits[col][1],
                                limits[row][0],
                                limits[row][1],
                            ),
                            origin="lower",
                            aspect="auto",
                        )
                    elif opts["upper"][n] == "contour":
                        if opts["contour_offdiag"]["percentile"]:
                            Z = probs2contours(Z, opts["contour_offdiag"]["levels"])
                        else:
                            Z = (Z - Z.min()) / (Z.max() - Z.min())
                        plt.contour(
                            X,
                            Y,
                            Z,
                            origin="lower",
                            extent=[
                                limits[col][0],
                                limits[col][1],
                                limits[row][0],
                                limits[row][1],
                            ],
                            colors=opts["samples_colors"][n],
                            levels=opts["contour_offdiag"]["levels"],
                        )
                    else:
                        pass
                elif opts["upper"][n] == "scatter":
                    plt.scatter(
                        v[:, col],
                        v[:, row],
                        color=opts["samples_colors"][n],
                        **opts["scatter_offdiag"],
                    )
                elif opts["upper"][n] == "plot":
                    plt.plot(
                        v[:, col],
                        v[:, row],
                        color=opts["samples_colors"][n],
                        **opts["plot_offdiag"],
                    )
                else:
                    pass

    return _arrange_plots(diag_func, upper_func, dim, limits, points, opts, fig=fig, axes=axes)


def plot_posterior_mapped_samples(
    posterior,
    x_o,
    true_theta=None,
    num_samples=20_000,
    sampling_device="cuda",
    show_progress_bars=False,
    original_limits=None,
    mapped_limits=None,
):
    font = {
        "weight": "bold",
        "size": 12,
    }
    mpl.rc("font", **font)
    mpl.rcParams["axes.linewidth"] = 1

    # copy x_o, true_theta
    x_o = x_o.clone()
    if true_theta.any() != None:
        true_theta = true_theta.clone()

    samples = sampling_from_posterior(
        sampling_device,
        posterior,
        x_o,
        num_samples=num_samples,
        show_progress_bars=show_progress_bars,
    )

    plot_limits = original_limits
    if mapped_limits != None:
        samples = convert_samples_range(samples, original_limits, mapped_limits)
        true_theta = convert_samples_range(true_theta, original_limits, mapped_limits)
        plot_limits = mapped_limits

    fig, ax = pairplot(
        samples,
        diag="kde",
        upper="kde",
        figsize=(10, 10),
        labels=["bias", "$\sigma^2_s$", "$\sigma^2_a$", "$\lambda$"],
        points=true_theta.cpu().numpy() if true_theta != None else None,
        points_colors="r",
        limits=plot_limits,
    )

    return fig, ax


def marginal_plot(
    samples,
    true_theta,
    origin_limits,
    dest_limits,
    moving_theta_idx=0,
    axes=None,
    credible_interval=95,
):
    num_params = len(dest_limits)
    # fig, axes = plt.subplots(1, num_params, figsize=(num_params * 6, 4))
    # fig.subplots_adjust(wspace=0.4)

    samples_dr = convert_samples_range(samples, origin_limits, dest_limits)
    true_theta_dr = convert_samples_range(true_theta, origin_limits, dest_limits)

    if credible_interval != 100:
        edge = (100 - credible_interval) / 2
        lower, upper = np.percentile(samples_dr, [edge, 100 - edge], axis=0)  # 95% interval
        # print(lower, upper)

    for i in range(num_params):
        ax = axes[i]
        density = gaussian_kde(samples_dr[:, i], bw_method="scott")
        xs = np.linspace(dest_limits[i][0], dest_limits[i][1], 100)
        ys = density(xs)
        ax.hist(samples_dr[:, i], bins=100, density=True, color="gray", alpha=0.8)
        if i == moving_theta_idx:
            ax.axvline(true_theta_dr[i], color="r", linewidth=4, linestyle="--")
        else:
            ax.axvline(true_theta_dr[i], color="r", linewidth=4)
        ax.plot(xs, ys, "k", linewidth=4)
        # ax.set_xlabel(prior_labels[i])
        ax.set_xlim(dest_limits[i][0], dest_limits[i][1])
        ax.grid(alpha=0.2)
        if i == 0:
            ax.set_ylabel("density")

        if credible_interval != 100:
            x_fill = np.linspace(lower[i], upper[i], 1000)
            y_fill = density(x_fill)
            ax.vlines(lower[i], 0, density(lower[i]), color="g", linewidth=1, linestyle="-")
            ax.vlines(upper[i], 0, density(upper[i]), color="g", linewidth=1, linestyle="-")
            ax.fill_between(x_fill, 0, y_fill, color="g", alpha=0.2)

    return axes
