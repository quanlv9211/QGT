import torch
import torch.nn as nn
from torch.nn import Parameter

from script.models.utils import get_activation
from script.manifolds.pseudohyperboloid import PseudoHyperboloid
from script.models.QGCN2.layers import PseudoGraphConvolution2
from script.models.GCN.layers import GraphConvolution
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter, scatter_add

class QGCN2(nn.Module):
    def __init__(self, args):
        super(QGCN2, self).__init__()
        self.device = args.device
        self.input_dim = args.nfeat
        self.in_time_dim = args.time_dim
        self.in_space_dim = args.space_dim
        self.out_time_dim = args.time_dim
        self.out_space_dim = args.space_dim   # this can be changed, but I let in_ _dim == out_ _dim for stable performance
        self.beta = nn.Parameter(torch.tensor(args.beta), requires_grad=False)
        args.nhid = args.time_dim + args.space_dim + 1
        self.hidden_dim = args.nhid
        self.output_dim = args.nout # nout = n_classes
        self.manifold = PseudoHyperboloid(self.in_time_dim, self.in_space_dim, self.beta)
        self.in_features = [self.in_time_dim, self.in_space_dim]
        self.out_features = [self.out_time_dim, self.out_space_dim]
        self.task = args.task

        self.linear = nn.Linear(self.input_dim, self.hidden_dim - 1, True)
        self.gcn = GCNConv(self.input_dim, self.hidden_dim - 1, normalize=False, bias=args.bias)
        #self.gcn = GraphConvolution(self.input_dim, self.hidden_dim - 1, get_activation(args.act), use_bias=args.bias, use_act=False)

        self.layer1 = PseudoGraphConvolution2(self.beta, self.in_features, self.out_features, dropout=args.dropout,
                                          act=get_activation(args.act), use_bias=args.bias)
        self.layer2 = PseudoGraphConvolution2(self.beta, self.out_features, self.out_features, dropout=args.dropout,
                                          act=get_activation(args.act), use_bias=args.bias)
        self.layer4 = PseudoGraphConvolution2(self.beta, self.out_features, self.out_features, dropout=args.dropout,
                                          act=get_activation(args.act), use_bias=args.bias)
        #self.layer5 = PseudoGraphConvolution2(self.beta, self.out_features, self.out_features, dropout=args.dropout,
        #                                  act=get_activation(args.act), use_bias=args.bias)
        #self.layer3 = nn.Linear(self.hidden_dim + 1, self.output_dim, False) 
        self.layer3 = nn.Linear(self.hidden_dim + 2, self.output_dim, False) #for residual
        self.feat = Parameter((torch.ones(args.num_nodes, self.input_dim)), requires_grad=True)
        #self.gcn2 = GCNConv(self.hidden_dim + 1, self.output_dim, normalize=False, bias=args.bias)
        #self.bias = nn.Parameter(torch.Tensor(self.output_dim), requires_grad=True)
        #self.weight = nn.Parameter(torch.Tensor(self.hidden_dim + 1, self.output_dim), requires_grad=True)
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.feat)
        glorot(self.layer3.weight)
        #zeros(self.layer3.bias)

    def residual_connection(self, y, x, time_dim):
        # x, y in Q^{p, q}
        x_ex = self.manifold.extrinsic_map(x)
        y_ex = self.manifold.extrinsic_map(y)
        xy_ex = self.manifold.inner(x_ex, y_ex, time_dim + 1).unsqueeze(1)
        xy_ex /= abs(self.beta)
        origin = x_ex.clone()
        origin[:,:] = 0
        origin[:,0] = self.beta.abs()**0.5
        y_ex = origin + y_ex
        y_ex *= xy_ex
        output = x_ex + y_ex
        return output # output in Q^{p, q + 1}

    def forward(self, edge_index, x=None):
        if x is None:
            x = self.feat
        #x_ = self.linear(x)
        assert not torch.isnan(x).any()
        x_ = self.gcn(x, edge_index.long())
        assert not torch.isnan(x_).any()
        # feature map
        o = torch.zeros_like(x_)
        x_t = torch.cat([o[:, 0:1], x_], dim=1)
        assert not torch.isnan(x_t).any()
        x_q = self.manifold.expmap0(x_t, self.beta)

        # forward
        h = self.layer1(x_q, edge_index)
        assert not torch.isnan(h).any()
        h = self.layer2(h, edge_index)
        #assert not torch.isnan(h).any()
        #h = self.layer4(h, edge_index)
        #h = self.layer5(h, edge_index)
        #h = self.residual_connection(x_q, h, self.out_time_dim)
        #assert not torch.isnan(h).any()
        if self.task == 'lp':
            return h
        h = self.manifold.logmap0(self.manifold.extrinsic_map(h), self.beta, self.out_time_dim + 1)
        assert not torch.isnan(h).any()
        #print(torch.min(h))
        #print(torch.max(h))
        output = self.layer3(h)

        # temporarily use normal Euclidean regression on tangent space at origin point.
        # tomorrow come up with better solution for regression
        return output

    def decoding_lp(self, z, edge_index):
        edge_i = edge_index[0]
        edge_j = edge_index[1]
        z_i = torch.nn.functional.embedding(edge_i, z)
        z_j = torch.nn.functional.embedding(edge_j, z)
        dist = self.manifold.sqdist(z_i, z_j, self.beta, self.out_time_dim)
        return dist

