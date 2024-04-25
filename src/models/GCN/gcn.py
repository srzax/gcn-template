import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, args, nfeat, nclass):
        super(GCN, self).__init__()
        self.args = args
        self.gc1 = GraphConvolution(nfeat, args.layers[0])
        self.gc2 = GraphConvolution(args.layers[1], nclass)
        self.dropout = args.dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.args.dropout)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)