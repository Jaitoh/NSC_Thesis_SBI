import matplotlib.pyplot as plt
cmaps = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def plot_a_mean_trace(a, probR, figure_name='', cmaps=cmaps):
    fig = plt.figure()
    # fig.suptitle('Model: ' + paramsFitted['allModelsList'][idx])
    plt.plot(a[::100], '.-', label=f'a1 probR={probR:.3f}', lw=2, color=cmaps[0])

    plt.xlabel('Time (sample)')
    plt.ylabel('a')

    lgd = plt.legend(loc = 'lower right', fontsize=24)
    # set the legend font to bold
    for text in lgd.get_texts():
        text.set_fontweight('bold')
    lgd.get_frame().set_facecolor('none')
    plt.grid(alpha=0.5)
    # change title font to bold
    plt.title(plt.title(figure_name).get_text(), fontsize=5)
    plt.show()

    return fig


def plot_parameters(ax, params, prior_min, prior_max, cmaps=cmaps):

    params = params[::-1]
    prior_min = prior_min[::-1]
    prior_max = prior_max[::-1]
    map_param = lambda x, pmin, pmax: (x-pmin)/(pmax-pmin)*2-1
    for i, (param) in enumerate(params):
        if i != 2:
            ax.hlines(y=i, xmin=-1, xmax=1, color='gray')
            ax.plot(-1, i, 'xk')
            ax.plot( 1, i, 'xk')
            # add text
            ax.text(-1.1, i, f'{prior_min[i]:.2f}', ha='right', va='center')
            ax.text( 1.1, i, f'{prior_max[i]:.2f}', ha='left', va='center')
            ax.plot(map_param(param, prior_min[i], prior_max[i]), i, 'o', color=cmaps[i])
            ax.text(map_param(param, prior_min[i], prior_max[i]), i-0.5, f'{param:.2f}', ha='center', va='center', color=cmaps[i])
        else:
            ax.plot(0, i, 'o', color=cmaps[i])
            ax.text(0, i-0.5, f'{param:.2f}', ha='center', va='center', color=cmaps[i])
        
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.set_xlim(-1.5, 1.5)
        ax.tick_params(labelbottom=False, labelleft=True)
        ax.set_yticks(list(range(5)))
        ax.set_yticklabels(['bias', 'sigma2a', 'sigma2i', 'sigma2s', 'L0'][::-1])
        ax.grid(alpha=0.1)