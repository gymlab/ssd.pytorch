import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
import math


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class MSGCN(nn.Module):
    def __init__(self, num_classes, num_scales, num_features, in_channel=300, t=0, p=1, adj_file=None):
        super(MSGCN, self).__init__()
        self.num_classes = num_classes
        self.num_scales = num_scales
        self.num_features = num_features

        self.gc_mid = GraphConvolution(in_channel, 1024)
        self.gc_weight = GraphConvolution(1024, 512)
        self.gc_bias = GraphConvolution(1024, 512)
        self.relu = nn.LeakyReLU(0.2)
        self.ms_aggregate_weight = nn.Linear(num_classes * num_scales, num_features)
        self.ms_aggregate_bias = nn.Linear(num_classes * num_scales, num_features)

        _adj = gen_adj(gen_A(num_classes * num_scales, t, p, adj_file))
        self.A = Parameter(_adj.float())
        # image normalization

    def forward(self, inp):
        inp = inp[0]
        inp = inp.repeat(1, self.num_scales)
        adj = self.A.detach()
        x = self.gc_mid(inp, adj)
        x = self.relu(x)
        weights = self.gc_weight(x, adj)
        biases = self.gc_bias(x, adj)

        weights = self.ms_aggregate_weight(weights.transpose(0, 1))
        biases = self.ms_aggregate_bias(biases.transpose(0, 1))

        return weights, biases


def gen_A(num_classes, t, p, adj_file):
    _adj = torch.load(adj_file).cuda()
    _nums = _adj.sum(dim=1)
    _nums = _nums.unsqueeze(1).expand_as(_adj)
    _adj = _adj / (_nums + 1e-6)
    _adj[_adj <= t] = 0
    _adj[_adj > t] = 1
    _adj = _adj * p / (_adj.sum(0, keepdim=True) + 1e-6)
    _adj = _adj + torch.eye(num_classes)
    print(_adj[i] for i in range(120))
    return _adj


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    # adj = adj.cpu().numpy()
    # for i in range(120):
    #     print(adj[i, :], sep='\n')
    return adj
