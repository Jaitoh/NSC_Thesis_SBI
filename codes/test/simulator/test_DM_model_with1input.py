debug = True # debug set to True to compare python and matlab results

# append sys.path
import sys
sys.path.append('./src')

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from dataset.model_sim_store_pR import plot_a_and_save
from simulator.DM_model import DM_model

cmaps = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple']
fig_dir = Path('../figures/model/comparison/Python')

# seqC1 = [0, -0.4, -0.4, -0.4, 0, 0.4, 0.4, 0.4, 0, -0.4, 0]
seqC1 = [0.00000,0.80000,-0.80000,0.00000,0.00000,-0.80000,-0.80000,-0.80000,0.00000,0.00000,-0.80000,0.80000,0.80000]
# [-0.72675, 28.77, 0, 5.0986, 6]
params = [-3.09194,20.02316,0.00000,1.46594,6.21906]

model_name = 'B-G-L0S-O-N-'
# input sequence plot
# fig = plt.figure(figsize=(8, 6))
# fig.suptitle('Input test sequences')
# plt.plot(seqC1, 'o-', label='seqC1', color=cmaps[0])
# plt.legend(loc='upper right')
# plt.grid(alpha=0.5)
# plt.savefig(Path(fig_dir/'input_sequence.png'))
# plt.xlabel('Time (sample)')
# plt.ylabel('Motion Strength')

model = DM_model(params=params, modelName=model_name)
a1, probR1 = model.simulate(seqC1)

fig = plot_a_and_save(a1, probR1,
                      figure_name='')
plt.show()
print(a1[::100])
# fig = plt.figure()
# # fig.suptitle('Model: ' + paramsFitted['allModelsList'][idx])
# plt.plot(a1[::100], '.-', label=f'a1 probR={probR1:.3f}', lw=2, color=cmaps[0])
#
# plt.xlabel('Time (sample)')
# plt.ylabel('Motion Strength')
#
# lgd = plt.legend(loc = 'lower right', fontsize=24)
# # set the legend font to bold
# for text in lgd.get_texts():
#     text.set_fontweight('bold')
# lgd.get_frame().set_facecolor('none')
# plt.grid(alpha=0.5)
# # change title font to bold
# plt.title(plt.title('Model: ' + paramsFitted['allModelsList'][idx]).get_text(), fontweight='bold', fontsize=24)
#
# # save the figure
# figPath = Path(f'{fig_dir}/fig_{model_type}.png')
# fig.savefig(figPath, dpi=300)
# # close the figure
# plt.close()
