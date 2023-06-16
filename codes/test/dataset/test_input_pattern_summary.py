import numpy as np
import pytest
import sys
sys.path.append('./src')
from simulator.seqC_pattern_summary import sequence_pattern_summary


# Test case 1: empty input
with pytest.raises(ValueError):
    sequence_pattern_summary()

# Test case 2: input sequence of shape (N, 15)
seqC = np.array([[0, 0.4, -0.4, 0.4, 0.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                 [0, -0.2, -0.2, -0.2, -0.2, 0, 0.2, 0.2, -0.2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
expected_output1 = np.array([[0.4   , 0.2857, 0.0714, 0.2142, 0.    , 0.    , 0.25  , 0.    , 0.25  , 0.25  ,0.25  ],
                             [0.2   , 0.5714, 0.3571, 0.1428, 0.0714, 0.375 , 0.125 , 0.125 , 0.125 , 0.    ,0.125 ]])
expected_output2 = np.array([[0.4, 1.0, 0.2857, 0.2857, 0.2857, 0.0, 0.1429, 0.1429],
                             [0.4, 2.0, 0.2857, 0.2857, 0.2857, 0.0, 0.2857, 0.0]])

np.testing.assert_array_almost_equal(sequence_pattern_summary(seqC), expected_output1)
np.testing.assert_array_almost_equal(sequence_pattern_summary(seqC, summaryType=2), expected_output2)

print('test pass for summaryType=1 and 2 with array inputs')
# Test case 3: dictionary input
# input_dict = {'pulse': np.random.randn(214214, 15),
#                 'dur': np.random.randn(214214, 1),
#                 'MS': np.random.randn(214214, 1),
#                 'nLeft': np.random.randn(214214, 1),
#                 'nRight': np.random.randn(214214, 1),
#                 'nPulse': np.random.randn(214214, 1),
#                 'hist_nLsame': np.random.randn(214214, 1),
#                 'hist_nLoppo': np.random.randn(214214, 1),
#                 'hist_nLelse': np.random.randn(214214, 1),
#                 'hist_nRsame': np.random.randn(214214, 1),
#                 'hist_nRoppo': np.random.randn(214214, 1),
#                 'hist_nRelse': np.random.randn(214214, 1),
#                 'chooseR': np.random.randn(214214, 1),
#                 'subjID': np.random.randn(214214, 1),
#                 'correct': np.random.randn(214214, 1)}
