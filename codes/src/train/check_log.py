from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
from matplotlib import pyplot as plt
import glob
import numpy as np
import cv2
import matplotlib as mpl
import argparse


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
    return ea_post,ea_val,ea_train

# load the event data
def get_event_data(ea_post, ea_val, ea_train):
    
    val_ = np.array([[e.wall_time, e.step, e.value] for e in ea_val.Scalars('log_probs')])
    val_time, val_step, val_log_probs = val_[:,0], val_[:,1], val_[:,2]

    train_ = np.array([[e.wall_time, e.step, e.value] for e in ea_train.Scalars('log_probs')])
    train_time, train_step, train_log_probs = train_[:,0], train_[:,1], train_[:,2]

    lr_ = np.array([[e.wall_time, e.step, e.value] for e in ea_post.Scalars('learning rate')])
    lr_time, lr_step, lr_lr = lr_[:,0], lr_[:,1], lr_[:,2]

    best_epochs = np.array([[e.wall_time, e.step, e.value] for e in ea_post.Scalars('run0/best_val_epoch_of_dset')])
    best_epochs_time, best_epochs_step, best_epochs_epoch = best_epochs[:,0], best_epochs[:,1], best_epochs[:,2]
    best_epochs_epoch = np.unique(best_epochs_epoch).astype(int)
    print(f'data: best epochs during training: {best_epochs_epoch}')
    
    return val_time,val_step,val_log_probs,train_step,train_log_probs,lr_step,lr_lr,best_epochs_epoch

# plot the learning rate and log_probs
def plot_lr_log_probs(val_time, val_step, val_log_probs, train_step, train_log_probs, lr_step, lr_lr, best_epochs_epoch):
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

# plot the posterior samples
def plot_posterior_samples(log_dir, best_epochs_epoch, exact_epoch=True):
    for case in ['train', 'val']:
        print(f'plotting posterior samples for {case}...', end=' ')
        fig, axs = plt.subplots(4, len(best_epochs_epoch), figsize=(len(best_epochs_epoch)*2, 4*2))
        # fig.subplots_adjust(hspace=0.1, wspace=0.1)

        for i in range(4):
            for j in range(len(best_epochs_epoch)):
                
                epoch = best_epochs_epoch[j] if exact_epoch else best_epochs_epoch[j] + 1
                # Load the image using OpenCV
                img = cv2.imread(f'{log_dir}/posterior/figures/posterior_x_{case}_{i}_epoch_{epoch}.png')

                # Check if the image has been loaded correctly
                if img is not None:
                    # Convert the image from BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Plot the image on the subplot
                    ax0 = axs[i, j]
                    ax0.imshow(img)
                    ax0.axis('off')
                    ax0.set_title(f'x_{case}_{i}\nepoch {epoch}')
                else:
                    print(f"Error: Unable to load image posterior_x_{case}_{i}_epoch_{epoch}.png")


        # save the figure
        plt.savefig(f'{log_dir}/posterior_samples_{case}.png', dpi=300)
        print(f'saved posterior samples to {log_dir}/posterior_samples_{case}.png')
        plt.close()


if __name__ == '__main__':

    log_dir_sample = '/home/wehe/tmp/NSC/codes/src/train/logs/train_L0/exp_b2_2_'
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--log_dir', type=str, default=log_dir_sample)
    argparser.add_argument('--exact_epoch', action='store_true')
    args = argparser.parse_args()


    log_dir = args.log_dir
    ea_post, ea_val, ea_train = get_events(log_dir)
    
    # get the event data
    val_time, val_step, val_log_probs, train_step, train_log_probs, lr_step, lr_lr, best_epochs_epoch = get_event_data(ea_post, ea_val, ea_train)
    
    # plot the learning rate and log_probs
    plot_lr_log_probs(val_time, val_step, val_log_probs, train_step, train_log_probs, lr_step, lr_lr, best_epochs_epoch)
    
    # plot the posterior samples
    plot_posterior_samples(log_dir, best_epochs_epoch, exact_epoch=args.exact_epoch)