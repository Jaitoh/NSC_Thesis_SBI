import cProfile
import pstats
import sys
sys.path.append('./src')
from simulator.DM_model import DM_model
import numpy as np

def test_DM_model():
    seqC = [0, -0.2, -0.2, 0.2, -0.2,  0.2,  0.2,    0, -0.2, 0.2, -0.2, 0,   -0.2, 0, -0.2]
    # modelName = 'B1G-L0S-O-N-'
    # params = [ 2, 10, 0, 10,  10, 10, 10]
    # model = DM_model(params=params, modelName=modelName)
    # model.simulate(seqC, debug=False)
    
    # modelName = 'B1G-L0S0O1N0'
    # params = [ 2, 10, 0, 10,  10, 10, 10, 2, 2, 1, 1]
    # model = DM_model(params=params, modelName=modelName)
    # model.simulate(seqC, debug=False)
    
    # boundary condition
    modelName = 'B0G0L0S0O1N0'
    params = [ 2, 10, 0, 1, 10, 10,1, 2, 2, 1, 1]
    # no boundary condition
    modelName = 'B-G0L0S0O1N0'
    params = [ 2, 10, 0, 1, 10,1, 2, 2, 1, 1]
    
    model = DM_model(params=params, modelName=modelName, cython=False)
    
    for i in range(100):
        a, probR = model.simulate(seqC, debug=False)
        # print(a[-10:], probR)
  
cProfile.run(statement='test_DM_model()')  # Replace with the name of your Python function
profiler = cProfile.Profile()
profiler.enable()
test_DM_model()
profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('tottime')
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()
# stats.dump_stats('./cProfile_DM_model.txt')
