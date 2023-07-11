from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
from matplotlib import pyplot as plt
import glob
import numpy as np
import cv2
import matplotlib as mpl
import argparse
import os
from tqdm import tqdm
import imageio

# plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 22
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 3


# load the event files
def get_events(log_dir):
    print(f"loading events from {log_dir}...", end=" ")

    post_path = glob.glob(f"{log_dir}/events.out.tfevents*")[0]
    val_path = glob.glob(f"{log_dir}/log_probs_validation/events.out.tfevents*")[0]
    train_path = glob.glob(f"{log_dir}/log_probs_training/events.out.tfevents*")[0]

    print(f"validation", end=" ")
    ea_val = EventAccumulator(str(val_path))
    ea_val.Reload()

    print(f"training", end=" ")
    ea_train = EventAccumulator(str(train_path))
    ea_train.Reload()

    print(f"posteriors", end=" ")
    ea_post = EventAccumulator(str(post_path))
    ea_post.Reload()

    print("done")

    # check if hist and images folders are available
    # if Path(f'{log_dir}/event_hist').exists():
    #     hist_path = glob.glob(f'{log_dir}/event_hist/events.out.tfevents*')[0]
    #     print(f'histograms', end=' ')
    #     ea_hist = EventAccumulator(str(hist_path))
    #     ea_hist.Reload()

    #     fig_path = glob.glob(f'{log_dir}/event_fig/events.out.tfevents*')[0]
    #     print(f'event figures', end=' ')
    #     ea_fig = EventAccumulator(str(fig_path))
    #     ea_fig.Reload()

    #     return ea_post,ea_val,ea_train, ea_hist, ea_fig
    # else:
    return ea_post, ea_val, ea_train


# load the event data
def get_event_data_p4(ea_post, ea_val, ea_train):
    val_perf = {}
    # print(f"==>> ea_post.Tags(): {ea_post.Tags()}")
    # print(f"==>> ea_train.Tags(): {ea_train.Tags()}")
    # print(f"==>> ea_val.Tags(): {ea_val.Tags()}")
    val_ = np.array(
        [[e.wall_time, e.step, e.value] for e in ea_val.Scalars("log_probs")]
    )
    val_perf["time"], val_perf["step"], val_perf["log_probs"] = (
        val_[:, 0],
        val_[:, 1],
        val_[:, 2],
    )

    train_perf = {}
    train_ = np.array(
        [[e.wall_time, e.step, e.value] for e in ea_train.Scalars("log_probs")]
    )
    train_perf["time"], train_perf["step"], train_perf["log_probs"] = (
        train_[:, 0],
        train_[:, 1],
        train_[:, 2],
    )

    lr = {}
    lr_ = np.array(
        [[e.wall_time, e.step, e.value] for e in ea_post.Scalars("learning_rates")]
    )
    lr["time"], lr["step"], lr["lr"] = lr_[:, 0], lr_[:, 1], lr_[:, 2]

    return val_perf, train_perf, lr


# plot the learning rate and log_probs
def plot_lr_log_probs_p4(val_perf, train_perf, lr, log_dir, exp_name):
    val_time, val_step, val_log_probs = (
        val_perf["time"],
        val_perf["step"],
        val_perf["log_probs"],
    )
    train_step, train_log_probs = train_perf["step"], train_perf["log_probs"]
    lr_step, lr_lr = lr["step"], lr["lr"]
    print("plotting learning rate and log_probs...", end=" ")

    plt.rcParams["figure.figsize"] = [25, 12]
    plt.rcParams["figure.autolayout"] = True

    ax = plt.GridSpec(2, 1)
    ax.update(wspace=0.1, hspace=0.4)

    ax0 = plt.subplot(ax[0])
    ax1 = plt.subplot(ax[1])

    ax0.plot(lr_step, lr_lr, "-", label="lr", lw=2)
    ax0.set_xlabel("epochs")
    ax0.set_ylabel("learning rate")
    ax0.grid(alpha=0.2)
    ax0.set_title(exp_name)

    plot_log_prob_p4(ax1, val_perf, train_perf, plot_time=True)

    # save the figure
    plt.savefig(f"{log_dir}/training_curve_.png")
    print(f"saved training curve to {log_dir}/training_curve_.png")


def plot_log_prob_p4(ax1, val_perf, train_perf, plot_time=True):
    val_time, val_step, val_log_probs = (
        val_perf["time"],
        val_perf["step"],
        val_perf["log_probs"],
    )
    train_step, train_log_probs = train_perf["step"], train_perf["log_probs"]

    ax1.plot(train_step, train_log_probs, "-", label="training", alpha=0.8, lw=2)
    ax1.plot(val_step, val_log_probs, "-", label="validation", alpha=0.8, lw=2)
    # mark the best validation epoch with v mark and text
    best_epoch = np.argmax(val_log_probs)
    ax1.plot(
        val_step[best_epoch],
        val_log_probs[best_epoch],
        "v",
        label="best epoch",
        alpha=0.8,
        lw=2,
    )
    ax1.text(
        val_step[best_epoch],
        val_log_probs[best_epoch],
        f"{val_log_probs[best_epoch]:.2f}",
        fontsize=12,
    )
    # upper = np.max(val_log_probs)
    # lower = np.percentile(val_log_probs, 10)
    # ax1.set_ylim(lower, upper)

    # ax1.legend(bbox_to_anchor=(1, 1), loc="upper left", borderaxespad=0.0)
    ax1.legend()
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("log_prob")
    ax1.grid(alpha=0.2)

    if plot_time:
        ax2 = ax1.twiny()
        ax2.plot(
            (val_time - val_time[0]) / 60 / 60,
            min(val_log_probs) * np.ones_like(val_log_probs),
            "-",
            alpha=0,
        )
        ax2.set_xlabel("time (hours)")
    return ax1


# plot the posterior samples
def plot_posterior_samples(log_dir, best_epochs_epoch, num_rows, exact_epoch=True):
    for case in ["train", "val"]:
        print(f"plotting posterior samples for {case}...", end=" ")
        fig, axs = plt.subplots(
            num_rows,
            len(best_epochs_epoch),
            figsize=(len(best_epochs_epoch) * 2, num_rows * 2),
        )
        # fig.subplots_adjust(hspace=0.1, wspace=0.1)

        for i in range(num_rows):
            for j in range(len(best_epochs_epoch)):
                epoch = (
                    best_epochs_epoch[j] if exact_epoch else best_epochs_epoch[j] + 1
                )
                # Load the image using OpenCV
                img = cv2.imread(
                    f"{log_dir}/posterior/figures/posterior_x_{case}_{i}_epoch_{epoch}.png"
                )

                # Check if the image has been loaded correctly
                if img is not None:
                    # Convert the image from BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Plot the image on the subplot
                    ax0 = axs[j] if num_rows == 1 else axs[i, j]
                    ax0.imshow(img)
                    ax0.axis("off")
                    ax0.set_title(f"x_{case}_{i}\nepoch {epoch}")
                else:
                    print(
                        f"Error: Unable to load image posterior_x_{case}_{i}_epoch_{epoch}.png"
                    )

        # save the figure
        plt.savefig(f"{log_dir}/posterior_samples_{case}.png", dpi=300)
        print(f"saved posterior samples to {log_dir}/posterior_samples_{case}.png")
        plt.close()


def load_img(img_path, ax, title):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def plot_one_img(
    chosen_plot_idx,
    plt_idx,
    log_dir,
    val_perf,
    train_perf,
    lr,
    exp_name,
):
    train_figure_names_0 = [
        f"posterior_seen_0_epoch_{idx}.png" for idx in chosen_plot_idx
    ]
    train_figure_names_1 = [
        f"posterior_seen_1_epoch_{idx}.png" for idx in chosen_plot_idx
    ]
    val_figure_names_0 = [
        f"posterior_unseen_0_epoch_{idx}.png" for idx in chosen_plot_idx
    ]
    val_figure_names_1 = [
        f"posterior_unseen_1_epoch_{idx}.png" for idx in chosen_plot_idx
    ]

    fig_dir = log_dir / "posterior" / "figures"

    # fig, axs = plt.subplots(5, 2, figsize=(20, 28))
    fig = plt.figure(figsize=(20, 28))
    # axs = axs.flatten()
    grid = plt.GridSpec(5, 2, wspace=0.1, hspace=0.5)
    ax0 = plt.subplot(grid[0, :])
    ax1 = plt.subplot(grid[1:3, 0])
    ax2 = plt.subplot(grid[1:3, 1])
    ax3 = plt.subplot(grid[3:5, 0])
    ax4 = plt.subplot(grid[3:5, 1])

    ax = ax1
    img_path = str(fig_dir / train_figure_names_0[plt_idx])
    title = f"seen 0 epoch {chosen_plot_idx[plt_idx]}"
    ax.set_title(title)
    load_img(img_path, ax, title)

    ax = ax2
    img_path = str(fig_dir / train_figure_names_1[plt_idx])
    title = f"seen 1 epoch {chosen_plot_idx[plt_idx]}"
    ax.set_title(title)
    load_img(img_path, ax, title)

    ax = ax3
    img_path = str(fig_dir / val_figure_names_0[plt_idx])
    title = f"unseen 0 epoch {chosen_plot_idx[plt_idx]}"
    ax.set_title(title)
    load_img(img_path, ax, title)

    ax = ax4
    img_path = str(fig_dir / val_figure_names_1[plt_idx])
    title = f"unseen 1 epoch {chosen_plot_idx[plt_idx]}"
    ax.set_title(title)
    load_img(img_path, ax, title)

    ax = ax0
    epoch_idx = chosen_plot_idx[plt_idx]
    # plot the training curve
    ax = plot_log_prob_p4(ax, val_perf, train_perf, plot_time=True)
    ax.plot(
        chosen_plot_idx,
        train_perf["log_probs"][chosen_plot_idx],
        "v",
        color="k",
        alpha=0.1,
    )
    ax.plot(epoch_idx, train_perf["log_probs"][epoch_idx], "v", color="green")
    ax.text(
        epoch_idx,
        min(train_perf["log_probs"]),
        f"{epoch_idx}",
        color="green",
        fontsize=8,
        ha="center",
        va="top",
    )

    ax.plot(
        chosen_plot_idx,
        val_perf["log_probs"][chosen_plot_idx],
        "v",
        color="k",
        alpha=0.1,
    )
    ax.plot(epoch_idx, val_perf["log_probs"][epoch_idx], "v", color="green")
    ax.text(
        epoch_idx,
        val_perf["log_probs"][epoch_idx] - 0.2,
        f"{val_perf['log_probs'][epoch_idx]:.2f}",
        color="orange",
        fontsize=8,
        ha="center",
        va="top",
    )
    ax.set_title(exp_name)

    # plot the learning rate
    ax0 = ax.twinx()
    lr_step, lr_lr = lr["step"], lr["lr"]
    ax0.plot(lr_step, lr_lr, "--", label="lr", lw=0.5, alpha=0.5, color="k")
    ax0.set_ylabel("learning rate")

    return fig


def get_plot_idx_p4(figures_all):
    plot_idx = []
    for fig in figures_all:
        fig = fig.split(".png")[0]
        # if fig.endswith("_shuffled"):
        plot_idx.append(int(fig.split("epoch_")[-1].split("_")[0]))
    return np.unique(plot_idx)


def animate_posterior(
    log_dir,
    num_frames,
    val_perf,
    train_perf,
    lr,
    duration=1000,
    exp_name="",
):
    # extract figure information
    figures_all = os.listdir(log_dir / "posterior/figures")
    unique_val = np.unique(
        [
            fig.split("unseen_")[1].split("_epoch")[0]
            for fig in figures_all
            if fig.startswith("posterior_unseen")
        ]
    )

    unique_train = np.unique(
        [
            fig.split("seen_")[1].split("_epoch")[0]
            for fig in figures_all
            if fig.startswith("posterior_seen")
        ]
    )
    num_val, num_train = len(unique_val), len(unique_train)
    print(
        f"# validation posterior tests: {num_val}\n# training posterior tests: {num_train}"
    )

    plot_idx = get_plot_idx_p4(figures_all)
    print(f"get {len(plot_idx)} figures: {plot_idx}")

    # generate animation
    images = []
    chosen_plot_idx = plot_idx[
        np.linspace(0, len(plot_idx) - 1, num_frames).astype(int)
    ]
    # add the last element for multiple times
    chosen_plot_idx = np.concatenate([chosen_plot_idx, [chosen_plot_idx[-1]] * 3])
    print(f"chosen {num_frames} plots to animate, idx: {chosen_plot_idx}")

    for plt_idx in tqdm(range(len(chosen_plot_idx))):
        fig = plot_one_img(
            chosen_plot_idx,
            plt_idx,
            log_dir,
            val_perf,
            train_perf,
            lr,
            exp_name,
        )
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)

    fig_name = f"posterior-{exp_name}.gif"
    imageio.mimsave(log_dir / fig_name, images, duration=duration, loop=0)
    print("saved animation to ", log_dir / fig_name)
    print()

    # save the last figure
    fig = plot_one_img(
        chosen_plot_idx,
        len(chosen_plot_idx) - 1,
        log_dir,
        val_perf,
        train_perf,
        lr,
        exp_name,
    )
    fig_name = f"posterior-{exp_name}.png"
    fig.savefig(log_dir / fig_name, dpi=300)
    print("saved figure to ", log_dir / fig_name)
    plt.close(fig)


if __name__ == "__main__":
    log_dir_sample = (
        "/home/ubuntu/tmp/NSC/codes/src/train/logs/train_L0_p4/p4-5Fs-1D-gru-mdn"
    )
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--log_dir", type=str, default=log_dir_sample)
    argparser.add_argument("--exp_name", type=str, default="exp-p2-3dur-test-0")
    argparser.add_argument("--num_frames", type=int, default=5)
    argparser.add_argument("--duration", type=int, default=1000)
    args = argparser.parse_args()

    log_dir = Path(args.log_dir)
    num_frames = args.num_frames
    duration = args.duration
    exp_name = args.exp_name

    # log_dir = Path(log_dir_sample)
    # num_frames = 5
    # duration = 1000
    # exp_name = "p4-5Fs-1D-gru-mdn"

    os.system(f"rm {log_dir}/training_curve_.png")
    os.system(f"rm {log_dir}/posterior_shuffled.gif")
    os.system(f"rm {log_dir}/posterior.gif")

    ea_post, ea_val, ea_train = get_events(log_dir)

    # extract data from event files
    val_perf, train_perf, lr = get_event_data_p4(ea_post, ea_val, ea_train)

    # plot training curves including the learning rate and log_probs
    plot_lr_log_probs_p4(val_perf, train_perf, lr, log_dir, exp_name)

    # animate posterior plots
    animate_posterior(
        log_dir,
        num_frames,
        val_perf,
        train_perf,
        lr,
        duration,
        exp_name,
    )

    # plot_shuffled = False
    # animate_posterior(log_dir, num_frames, plot_shuffled,
    #                 val_perf, train_perf, lr, best, duration, exp_name)

    # remove event_fig files
    os.system(f"rm -r {log_dir}/event_fig/")
    print("removed event_fig files")
