debug = True # debug set to True to compare python and matlab results

# append sys.path
import sys
sys.path.append('./src')

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from parse_data.decode_parameter import decode_mat_fitted_parameters
from simulator.DM_model import DM_model

cmaps = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple']
fig_dir = Path('../figures/model/comparison/Python')

# load .mat file
filePath = Path('../data/params/263 models fitPars/data_fitPars_S1.mat')
paramsFitted = decode_mat_fitted_parameters(filePath)

seqC1 = [0, 0.4, -0.4]
seqC2 = [0, 0.1, -0.1]
seqC3 = [0, -0.2, -0.2, 0.2, -0.2,  0.2,  0.2,    0, -0.2, 0.2, -0.2, 0,   -0.2, 0, -0.2]
seqC4 = [0, 0.1,  0,    0.1,  0.1, -0.1, -0.1, -0.1,  0.1, 0,   -0.1, 0.1,    0, 0, 0   ]

# Model types and corresponding indices
model_types = {
    'G2': 133-1,
    'G6': 137-1,
    'L2': 138-1,
    'L6': 142-1,
    'S0x0x0': 143-1,
    'S1x1x2': 161-1,
    'G3L2': 132-1,
    'B0': 212-1,
    'B1G1L1S1x1': 241-1
}

# input sequence plot
fig = plt.figure(figsize=(8, 6))
fig.suptitle('Input test sequences')
plt.plot(seqC1, 'o-', label='seqC1', color=cmaps[0])
plt.plot(seqC2, 'o-', label='seqC2', color=cmaps[1])
plt.plot(seqC3, 'o-', label='seqC3', color=cmaps[2])
plt.plot(seqC4, 'o-', label='seqC4', color=cmaps[3])
plt.legend(loc='upper right')
plt.grid(alpha=0.5)
plt.savefig(Path(fig_dir/'input_sequence.png'))
plt.xlabel('Time (sample)')
plt.ylabel('Motion Strength')


for model_type, idx in model_types.items():
    params = {}
    params['bias']   = paramsFitted['bias'][idx]
    params['sigmas'] = paramsFitted['sigmas'][idx,:]
    params['BGLS']   = paramsFitted['BGLS'][idx, :, :]
    params['modelName'] = paramsFitted['allModelsList'][idx]
    print('Model: ' + paramsFitted['allModelsList'][idx])

    model = DM_model(params=params)
    a1, probR1 = model.stoch_simulation(seqC1, debug=debug)
    a2, probR2 = model.stoch_simulation(seqC2, debug=debug)
    a3, probR3 = model.stoch_simulation(seqC3, debug=debug)
    a4, probR4 = model.stoch_simulation(seqC4, debug=debug)

    print(probR1, probR2, probR3, probR4)

    fig = plt.figure()
    # fig.suptitle('Model: ' + paramsFitted['allModelsList'][idx])
    plt.plot(a1[::100], '.-', label=f'a1 probR={probR1:.3f}', lw=2, color=cmaps[0])
    plt.plot(a2[::100], '.-', label=f'a2 probR={probR2:.3f}', lw=2, color=cmaps[1])
    plt.plot(a3[::100], '.-', label=f'a3 probR={probR3:.3f}', lw=2, color=cmaps[2])
    plt.plot(a4[::100], '.-', label=f'a4 probR={probR4:.3f}', lw=2, color=cmaps[3])
    
    plt.xlabel('Time (sample)')
    plt.ylabel('Motion Strength')

    lgd = plt.legend(loc = 'lower right', fontsize=24)
    # set the legend font to bold
    for text in lgd.get_texts():
        text.set_fontweight('bold')
    lgd.get_frame().set_facecolor('none')
    plt.grid(alpha=0.5)
    # change title font to bold
    plt.title(plt.title('Model: ' + paramsFitted['allModelsList'][idx]).get_text(), fontweight='bold', fontsize=24)
    
    # save the figure
    figPath = Path(f'{fig_dir}/fig_{model_type}.png')
    fig.savefig(figPath, dpi=300)
    # close the figure
    plt.close()

# generate a plot for illustrating the output mean trace of the model
fig = plt.figure()
plt.plot(a4[::100], '.-', label=f'a4 probR={probR4:.3f}', lw=2, color='k')
# remove axis
plt.axis('off')
# remove the frame of the figure
plt.gca().spines['top'].set_visible(False)
# set background color to transparent
plt.gca().set_facecolor('none')

# save the figure
figPath = Path(f'{fig_dir}/fig_{model_type}_demo.png')
fig.savefig(figPath, dpi=300)
# close the figure
plt.close()

