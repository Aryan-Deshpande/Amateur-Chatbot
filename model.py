import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, insize, hsize, nclasses):
        super(NeuralNet, self).__init__()
        # feed forward neural network
        # 2 hidden layers
        self.l1 = nn.Linear(insize, hsize)
        self.l2 = nn.Linear(hsize, hsize)
        self.l2 = nn.Linear(hsize, hsize)
        self.l3 = nn.Linear(hsize, nclasses)

        self.reLU = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.reLU(out)

        out = self.l2(out)
        out = self.reLU(out)

        out = self.l3(out)
        out = self.reLU(out)

        # no activation , no softmax
        return out