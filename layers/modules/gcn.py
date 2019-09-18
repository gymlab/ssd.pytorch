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
        # self.bn = nn.BatchNorm1d(out_features)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        # output = self.bn(output)
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

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 1024)
        self.gc3 = GraphConvolution(1024, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.ms_aggregate_attention = nn.Sequential(
            nn.Linear(1024 * 20, 512 + 1024 + 512 + 256 + 256 + 256),
            nn.Sigmoid())

        _adj = gen_adj(gen_A(num_classes, t, p, adj_file))
        self.A = Parameter(_adj.float())
        # image normalization

    def forward(self, inp):
        inp = inp.repeat(1, 1)    # (20, 300)
        adj = self.A.detach()       # (20, 20)
        x = self.relu(self.gc1(inp, adj))   # [20, 1024]
        attention = self.relu(self.gc2(x, adj))    # [20, 1024]
        attention = self.relu(self.gc3(attention, adj))

        attention = self.ms_aggregate_attention(attention.view(1, -1))     # (1, total_channels)

        return attention.squeeze()


def gen_A(num_classes, t, p, adj_file):
    _adj = torch.load(adj_file).cuda()
    _nums = _adj.sum(dim=1)
    _nums = _nums.unsqueeze(1).expand_as(_adj)
    _adj = _adj / (_nums + 1e-6)
    _adj[_adj <= t] = 0
    _adj[_adj > t] = 1
    _adj = _adj * p / (_adj.sum(0, keepdim=True) + 1e-6)
    _adj = _adj + torch.eye(num_classes)
    # print(_adj[i] for i in range(120))
    return _adj


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    # adj = adj.cpu().numpy()
    # for i in range(20):
    #     print(adj[i, :], sep='\n')
    return adj
