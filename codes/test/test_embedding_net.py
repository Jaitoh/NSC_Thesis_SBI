import sys
sys.path.append('./src')
from nerual_nets import *
from torchsummary import summary

# test network with summary
net = RNN_Multi_Head(700, 16, 16)
summary(net, (700, 16, 16))
