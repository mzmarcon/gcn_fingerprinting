import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.init import xavier_uniform_, zeros_


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        xavier_uniform_(self.weight)
        zeros_(self.bias)        

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class ChebyGraphConvolution(Module):

    def __init__(self, in_features, out_features, max_deg=3, bias=True):
        super(ChebyGraphConvolution, self).__init__()
        self.cheby_deg = 1 + max_deg
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(self.cheby_deg,in_features,out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)        
        self.reset_parameters()

    def reset_parameters(self):
        xavier_uniform_(self.weight)
        zeros_(self.bias)

    def forward(self, input, adj):
        supports = list()
        pre_outs = list()
        for i in range(self.cheby_deg):
            supports.append(torch.mm(input,self.weight[i]))
            pre_outs.append(torch.spmm(adj[i], supports[i]))
        output = sum(pre_outs)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
