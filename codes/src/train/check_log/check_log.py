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

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 10}

mpl.rc('font', **font)


# load the event files
def get_events(log_dir):
    print(f'loading events from {log_dir}...', end=' ')
    
    post_path = glob.glob(f'{log_dir}/events.out.tfevents*')[0]
    val_path = glob.glob(f'{log_dir}/log_probs_validation/events.out.tfevents*')[0]
    train_path = glob.glob(f'{log_dir}/log_probs_training/events.out.tfevents*')[0]

    print(f'validation', end=' ')
    ea_val = EventAccumulator(str(val_path))
    ea_val.Reload()

    print(f'training', end=' ')
    ea_train = EventAccumulator(str(train_path))
    ea_train.Reload()
    
    print(f'posteriors', end=' ')
    ea_post = EventAccumulator(str(post_path))
    ea_post.Reload()
    
    print('done')
    
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
    return ea_post,ea_val,ea_train

# load the event data
def get_event_data(ea_post, ea_val, ea_train):
    
    val_perf = {}
    val_ = np.array([[e.wall_time, e.step, e.value] for e in ea_val.Scalars('log_probs')])
    val_perf['time'], val_perf['step'], val_perf['log_probs'] = val_[:,0], val_[:,1], val_[:,2]
    
    train_perf = {}
    train_ = np.array([[e.wall_time, e.step, e.value] for e in ea_train.Scalars('log_probs')])
    train_perf['time'], train_perf['step'], train_perf['log_probs'] = train_[:,0], train_[:,1], train_[:,2]
    
    lr = {}
    lr_ = np.array([[e.wall_time, e.step, e.value] for e in ea_post.Scalars('learning_rates')])
    lr['time'], lr['step'], lr['lr'] = lr_[:,0], lr_[:,1], lr_[:,2]

    # best_epochs = np.array([[e.wall_time, e.step, e.value] for e in ea_post.Scalars('run0/best_val_epoch_of_dset')])
    # best_epochs = np.array([[e.wall_time, e.step, e.value] for e in ea_post.Scalars('run0/best_val_epoch')])
    # best_epochs_time, best_epochs_step, best_epochs_epoch = best_epochs[:,0], best_epochs[:,1], best_epochs[:,2]
    # best_epochs_epoch = np.unique(best_epochs_epoch).astype(int)
    # print(f'data: best epochs during training: {best_epochs_epoch}')
    best = {}
    best_epochs = np.array([[e.wall_time, e.step, e.value] for e in ea_post.Scalars('run0/best_val_epoch_board')])
    best['time'], best['step'], best['epoch_num'] = best_epochs[-1]
    
    return val_perf, train_perf, lr, best

# plot the learning rate and log_probs
def plot_lr_log_probs_(val_time, val_step, val_log_probs, train_step, train_log_probs, lr_step, lr_lr, best_epochs_epoch):
    print('plotting learning rate and log_probs...', end=' ')
    
    plt.rcParams["figure.figsize"] = [10, 8]
    plt.rcParams["figure.autolayout"] = True

    ax = plt.GridSpec(2, 1)
    ax.update(wspace=0.1, hspace=0.4)

    ax0 = plt.subplot(ax[0])
    ax1 = plt.subplot(ax[1])

    ax0.plot(lr_step, lr_lr, '-', label='lr', lw=2)
    ax0.plot(best_epochs_epoch, lr_lr[list(best_epochs_epoch)], 'v', color='tab:red', lw=2)

    ax0.set_xlabel('epochs')
    ax0.set_ylabel('learning rate')
    ax0.grid(alpha=0.2)
    ax0.set_title('training curve')

    ax1.plot(train_step, train_log_probs, '-', label='training', alpha=0.8, lw=2)
    ax1.plot(val_step, val_log_probs, '-', label='validation', alpha=0.8, lw=2)
    ax1.plot(best_epochs_epoch, val_log_probs[list(best_epochs_epoch)], 'v', color='red', lw=2)
    for i in range(len(best_epochs_epoch)):
        ax1.text(best_epochs_epoch[i], val_log_probs[best_epochs_epoch[i]]+0.02, f'{val_log_probs[best_epochs_epoch[i]]:.2f}', color='red', fontsize=10, ha='center', va='bottom')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('log_prob')
    ax1.grid(alpha=0.2)

    ax2 = ax1.twiny()
    ax2.plot((val_time-val_time[0])/60/60, min(val_log_probs)*np.ones_like(val_log_probs), '-', alpha=0)
    ax2.set_xlabel('time (hours)')

    # save the figure
    plt.savefig(f'{log_dir}/training_curve.png')
    print(f'saved training curve to {log_dir}/training_curve.png')
    plt.close()

def plot_lr_log_probs(val_perf, train_perf, lr, best, log_dir):
    
    val_time, val_step, val_log_probs = val_perf['time'], val_perf['step'], val_perf['log_probs']
    train_step, train_log_probs = train_perf['step'], train_perf['log_probs']
    lr_step, lr_lr = lr['step'], lr['lr']
    best_epochs_epoch = best['epoch_num']
    print('plotting learning rate and log_probs...', end=' ')
    
    plt.rcParams["figure.figsize"] = [10, 8]
    plt.rcParams["figure.autolayout"] = True

    ax = plt.GridSpec(2, 1)
    ax.update(wspace=0.1, hspace=0.4)

    ax0 = plt.subplot(ax[0])
    ax1 = plt.subplot(ax[1])

    ax0.plot(lr_step, lr_lr, '-', label='lr', lw=2)
    if isinstance(best_epochs_epoch, float):
        ax0.plot(best_epochs_epoch, lr_lr[int(best_epochs_epoch)], 'v', color='tab:red', lw=2)
    else:
        ax0.plot(best_epochs_epoch, lr_lr[list(best_epochs_epoch)], 'v', color='tab:red', lw=2)

    ax0.set_xlabel('epochs')
    ax0.set_ylabel('learning rate')
    ax0.grid(alpha=0.2)
    ax0.set_title('training curve')

    plot_log_prob(ax1, val_perf, train_perf, lr, best, plot_time=True)

    # save the figure
    plt.savefig(f'{log_dir}/training_curve_.png')
    print(f'saved training curve to {log_dir}/training_curve.png')

def plot_log_prob(ax1, val_perf, train_perf, lr, best, plot_time=True):
    
    val_time, val_step, val_log_probs = val_perf['time'], val_perf['step'], val_perf['log_probs']
    train_step, train_log_probs = train_perf['step'], train_perf['log_probs']
    lr_step, lr_lr = lr['step'], lr['lr']
    best_epochs_epoch = best['epoch_num']
    
    ax1.plot(train_step, train_log_probs, '-', label='training', alpha=0.8, lw=2)
    ax1.plot(val_step, val_log_probs, '-', label='validation', alpha=0.8, lw=2)
    if isinstance(best_epochs_epoch, float):
        ax1.plot(best_epochs_epoch, val_log_probs[int(best_epochs_epoch)], 'v', color='red', lw=2)
        ax1.text(best_epochs_epoch, val_log_probs[int(best_epochs_epoch)]+0.02, f'{val_log_probs[int(best_epochs_epoch)]:.2f}', color='red', fontsize=10, ha='center', va='bottom')
    else:
        ax1.plot(best_epochs_epoch, val_log_probs[list(best_epochs_epoch)], 'v', color='red', lw=2)
        for i in range(len(best_epochs_epoch)):
            ax1.text(best_epochs_epoch[i], val_log_probs[best_epochs_epoch[i]]+0.02, f'{val_log_probs[best_epochs_epoch[i]]:.2f}', color='red', fontsize=10, ha='center', va='bottom')
        
    ax1.legend(loc='best')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('log_prob')
    ax1.grid(alpha=0.2)

    if plot_time:
        ax2 = ax1.twiny()
        ax2.plot((val_time-val_time[0])/60/60, min(val_log_probs)*np.ones_like(val_log_probs), '-', alpha=0)
        ax2.set_xlabel('time (hours)')
    return ax1

# plot the posterior samples
def plot_posterior_samples(log_dir, best_epochs_epoch, num_rows, exact_epoch=True):
    for case in ['train', 'val']:
        print(f'plotting posterior samples for {case}...', end=' ')
        fig, axs = plt.subplots(num_rows, len(best_epochs_epoch), figsize=(len(best_epochs_epoch)*2, num_rows*2))
        # fig.subplots_adjust(hspace=0.1, wspace=0.1)

        for i in range(num_rows):
            for j in range(len(best_epochs_epoch)):
                
                epoch = best_epochs_epoch[j] if exact_epoch else best_epochs_epoch[j] + 1
                # Load the image using OpenCV
                img = cv2.imread(f'{log_dir}/posterior/figures/posterior_x_{case}_{i}_epoch_{epoch}.png')

                # Check if the image has been loaded correctly
                if img is not None:
                    # Convert the image from BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Plot the image on the subplot
                    ax0 = axs[j] if num_rows == 1 else axs[i, j]
                    ax0.imshow(img)
                    ax0.axis('off')
                    ax0.set_title(f'x_{case}_{i}\nepoch {epoch}')
                else:
                    print(f"Error: Unable to load image posterior_x_{case}_{i}_epoch_{epoch}.png")


        # save the figure
        plt.savefig(f'{log_dir}/posterior_samples_{case}.png', dpi=300)
        print(f'saved posterior samples to {log_dir}/posterior_samples_{case}.png')
        plt.close()

def plot_one_img(chosen_plot_idx, plt_idx, plot_shuffled,
                 log_dir, val_perf, train_perf, lr, best, exp_name):
    
    if plot_shuffled:
        train_figure_names_0 = [f'posterior_x_train_0_epoch_{idx}_shuffled.png' for idx in chosen_plot_idx]
        train_figure_names_1 = [f'posterior_x_train_1_epoch_{idx}_shuffled.png' for idx in chosen_plot_idx]
        val_figure_names_0   = [f'posterior_x_val_0_epoch_{idx}_shuffled.png'   for idx in chosen_plot_idx]
        val_figure_names_1   = [f'posterior_x_val_1_epoch_{idx}_shuffled.png'   for idx in chosen_plot_idx]
    else:
        train_figure_names_0 = [f'posterior_x_train_0_epoch_{idx}.png' for idx in chosen_plot_idx]
        train_figure_names_1 = [f'posterior_x_train_1_epoch_{idx}.png' for idx in chosen_plot_idx]
        val_figure_names_0   = [f'posterior_x_val_0_epoch_{idx}.png'   for idx in chosen_plot_idx]
        val_figure_names_1   = [f'posterior_x_val_1_epoch_{idx}.png'   for idx in chosen_plot_idx]

    fig = plt.figure(figsize=(3*2, 2*4))

    ax = plt.subplot(3,2,3)
    img = cv2.imread(str(log_dir/'posterior'/'figures'/train_figure_names_0[plt_idx]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    title = 'train x_0 shuffled' if plot_shuffled else 'train x_0' 
    ax.set_title(title)

    ax = plt.subplot(3,2,4)
    img = cv2.imread(str(log_dir/'posterior'/'figures'/train_figure_names_1[plt_idx]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    title = 'train x_1 shuffled' if plot_shuffled else 'train x_1' 
    ax.set_title(title)

    ax = plt.subplot(3,2,5)
    img = cv2.imread(str(log_dir/'posterior'/'figures'/val_figure_names_0[plt_idx]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    title = 'val x_0 shuffled' if plot_shuffled else 'val x_0' 
    ax.set_title(title)

    ax = plt.subplot(3,2,6)
    img = cv2.imread(str(log_dir/'posterior'/'figures'/val_figure_names_1[plt_idx]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    title = 'val x_1 shuffled' if plot_shuffled else 'val x_1' 
    ax.set_title(title)

    epoch_idx = chosen_plot_idx[plt_idx]
    
    ax = plt.subplot(3,1,1) # plot the training curve
    ax = plot_log_prob(ax, val_perf, train_perf, lr, best, plot_time=True)
    ax.plot(chosen_plot_idx, train_perf['log_probs'][chosen_plot_idx], 'v', color='grey', alpha=0.2)
    ax.plot(epoch_idx, train_perf['log_probs'][epoch_idx], 'v', color='green')
    ax.text(epoch_idx, min(train_perf['log_probs']), f"{epoch_idx}", color='green', fontsize=8, ha='center', va='top')
    
    ax.plot(chosen_plot_idx, val_perf['log_probs'][chosen_plot_idx], 'v', color='grey', alpha=0.2)
    ax.plot(epoch_idx, val_perf['log_probs'][epoch_idx], 'v', color='green')
    ax.text(epoch_idx, val_perf['log_probs'][epoch_idx]-0.2, f"{val_perf['log_probs'][epoch_idx]:.2f}", color='orange', fontsize=8, ha='center', va='top')
    ax.set_title(exp_name)
    
    return fig

def get_plot_idx(figures_all):
    plot_idx = []
    for fig in figures_all:
        fig = fig.split('.png')[0]
        if fig.endswith('_shuffled'):
            plot_idx.append(int(fig.split('epoch_')[-1].split('_')[0]))
    plot_idx = np.unique(plot_idx)
    return plot_idx

def animate_posterior(log_dir, num_frames, plot_shuffled, 
                      val_perf, train_perf, lr, best, duration=1000, exp_name=''):
    # extract figure information
    figures_all = os.listdir(log_dir/'posterior/figures')
    unique_val = np.unique([fig.split('x_val_')[1].split('_epoch')[0] for fig in figures_all if fig.startswith('posterior_x_val')])

    unique_train = np.unique([fig.split('x_train_')[1].split('_epoch')[0] for fig in figures_all if fig.startswith('posterior_x_train')])
    num_val, num_train = len(unique_val), len(unique_train)
    print(f'# validation posterior tests: {num_val}\n# training posterior tests: {num_train}')

    best_epoch = int(best['epoch_num'])
    plot_idx = get_plot_idx(figures_all)
    print(f'best epoch: {best_epoch}\nget {len(plot_idx)} figures: {plot_idx}')

    # generate animation
    images = []
    chosen_plot_idx = plot_idx[np.linspace(0, len(plot_idx)-1, num_frames).astype(int)]
    print(f'chosen {num_frames} plots to animate, idx: {chosen_plot_idx}')

    for plt_idx in tqdm(range(len(chosen_plot_idx))):
        fig = plot_one_img(chosen_plot_idx, plt_idx, plot_shuffled,
                        log_dir, val_perf, train_perf, lr, best, exp_name)
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)

    fig_name = 'posterior_shuffled.gif' if plot_shuffled else 'posterior.gif'
    imageio.mimsave(log_dir/fig_name, images, duration=duration, loop=0)
    print('saved animation to ', log_dir/fig_name)
    print()


if __name__ == '__main__':

    log_dir_sample = '/home/wehe/tmp/NSC/codes/src/train/logs/train_L0/exp-p2-3dur-test-0'
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--log_dir', type=str, default=log_dir_sample)
    argparser.add_argument('--exp_name', type=str, default='exp-p2-3dur-test-0')
    argparser.add_argument('--num_frames', type=int, default=5)
    argparser.add_argument('--duration', type=int, default=1000)
    
    
    args = argparser.parse_args()

    log_dir     = Path(args.log_dir)
    num_frames  = args.num_frames
    duration    = args.duration
    exp_name    = args.exp_name
    
    ea_post, ea_val, ea_train = get_events(log_dir)
    
    # extract data from event files
    val_perf, train_perf, lr, best = get_event_data(ea_post, ea_val, ea_train)

    # plot training curves including the learning rate and log_probs
    plot_lr_log_probs(val_perf, train_perf, lr, best, log_dir)

    # animate posterior plots
    plot_shuffled = True
    animate_posterior(log_dir, num_frames, plot_shuffled, 
                    val_perf, train_perf, lr, best, duration, exp_name)

    # animate posterior plots
    plot_shuffled = False
    animate_posterior(log_dir, num_frames, plot_shuffled, 
                    val_perf, train_perf, lr, best, duration, exp_name)