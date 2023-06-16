import sys
sys.path.append('./src')
from neural_nets.embedding_nets import RNN_Multi_Head
from torchsummary import summary

# test network with summary
DM = 21
S = 700
L = 16
net = RNN_Multi_Head(DM, S, L).cuda()
summary(net, (DM, S, L))

print(net)