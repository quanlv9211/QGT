import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot, zeros

from script.models.utils import get_activation
from script.manifolds.pseudohyperboloid import PseudoHyperboloid
from script.models.QGCN2.layers import PseudoGraphConvolution2, PseudoLinear2, PseudoGraphConvolution3, PseudoAct2, PseudoAgg2
from script.models.QGT.layers import PseudoLayerNorm2, PseudoTransConv, PseudoDropout2

class QGT(nn.Module):
    def __init__(self, args):
        super(QGT, self).__init__()
        self.device = args.device
        self.input_dim = args.nfeat
        self.in_time_dim = args.time_dim # q
        self.in_space_dim = args.space_dim # p
        self.out_time_dim = args.time_dim
        self.out_space_dim = args.space_dim   # this can be changed, but I let in_ _dim == out_ _dim for stable performance
        self.beta = torch.tensor(args.beta).to(args.device)
        args.nhid = args.time_dim + args.space_dim + 1
        self.hidden_dim = args.nhid
        self.output_dim = args.nout # nout = n_classes
        self.manifold = PseudoHyperboloid(self.in_time_dim, self.in_space_dim, self.beta)
        self.in_features = [self.in_time_dim, self.in_space_dim]
        self.out_features = [self.out_time_dim, self.out_space_dim]
        self.dropout_time = args.dropout_time
        self.dropout_space = args.dropout_space
        self.g_dropout_time = args.g_dropout_time
        self.g_dropout_space = args.g_dropout_space
        self.graph_num_layers = args.graph_num_layers
        self.graph_dropout = args.graph_dropout
        self.using_pretrained_feat = args.using_pretrained_feat
        self.task = args.task

        self.first_linear = nn.Linear(self.input_dim, self.hidden_dim - 1, True)
        #self.gcn = GCNConv(self.input_dim, self.hidden_dim - 1, normalize=False, bias=args.bias)

        if self.using_pretrained_feat:
            self.feat = Parameter((torch.ones(args.num_nodes, self.input_dim)), requires_grad=True)

        self.num_layers = args.trans_num_layers

        self.trans_conv = PseudoTransConv(self.manifold, self.beta, self.in_features, self.out_features,
                                            self.out_features, get_activation(args.act), args)

        self.graph_convs = nn.ModuleList()
        self.graph_dropouts = nn.ModuleList()
        self.in_features_g = [self.in_time_dim, self.in_space_dim]
        self.out_features_g = [self.out_time_dim, self.out_space_dim]
        for i in range(self.graph_num_layers):
            if i < self.graph_num_layers - 1:
                self.graph_convs.append(PseudoGraphConvolution2(self.beta, self.in_features_g, self.out_features_g, dropout=0.0,
                                                 act=get_activation(args.act), use_bn=True))
            else:
                self.graph_convs.append(PseudoGraphConvolution2(self.beta, self.out_features_g, self.out_features_g, dropout=0.0,
                                                 act=get_activation(args.act), use_bn=False))
            self.graph_dropouts.append(PseudoDropout2(self.out_time_dim, self.out_space_dim, self.beta, self.g_dropout_time, 
                                        self.g_dropout_space))


        #self.last_gcn = PseudoGraphConvolution2(self.beta, self.out_features_g, self.out_features_g, dropout=0.0,
        #                                         act=get_activation(args.act), use_bn=False)
        
        #self.last_dropout = PseudoDropout2(self.out_time_dim + 1, self.out_space_dim, self.beta, dropout_time=self.dropout_time, dropout_space=self.dropout_space)
        self.use_graph = args.use_graph
        self.graph_weight = args.graph_weight
        if self.use_graph:
            self.last_bn = PseudoLayerNorm2(self.out_time_dim + 1, self.out_space_dim, self.beta)
            self.last_layer = nn.Linear(self.hidden_dim + 2, self.output_dim, False)
        else:
            self.last_layer = nn.Linear(self.hidden_dim + 1, self.output_dim, False)

        self.reset_parameters()
        
        # params1: trans
        # params2: gcn
        # params3: other
        self.params3 = (list(self.last_layer.parameters()))
        #self.params3.extend(list(self.first_linear.parameters()))
        #self.params3.extend(list(self.last_gcn.parameters()))

        self.params2 = (list(self.graph_convs.parameters()) if self.graph_convs is not None else [])
        #self.params2.extend(list(self.last_gcn.parameters()))

        self.params1 = list(self.trans_conv.parameters())
        self.params1.extend(list(self.first_linear.parameters()))
        #self.params1.extend(list(self.last_gcn.parameters()))

        if self.using_pretrained_feat:
            self.params3.extend(list(self.feat))

        self.graph_conv_clip_config = ["graph_convs"] # can xem lai
        self.first_layer_clip_config = ["first_linear"] # can xem lai

    def reset_parameters(self):
        glorot(self.last_layer.weight)
        #glorot(self.linear.weight)
        #zeros(self.linear.bias)

    
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

    def mid_point(self, x, y, time_dim, weight):
        # x, y in Q^{p, q}
        x_ex = self.manifold.extrinsic_map(x)
        y_ex = self.manifold.extrinsic_map(y)
        x_ex = self.manifold.logmap0(x_ex, self.beta, time_dim + 1)
        y_ex = self.manifold.logmap0(y_ex, self.beta, time_dim + 1)
        x_ex *= weight
        y_ex *= (1 - weight)
        z = y_ex + x_ex
        output = self.manifold.expmap0(z, self.beta, time_dim + 1)
        return output # output in Q^{p, q + 1}

    def forward(self, edge_index, degrees, x=None):
        if self.using_pretrained_feat:
            x = self.feat
        #x_t = self.first_linear(x)
        x_t = self.gcn(x, edge_index.long())

        # feature map
        o = torch.zeros_like(x_t)
        x_t_tangent = torch.cat([o[:, 0:1], x_t], dim=1)
        x_q_t = self.manifold.expmap0(x_t_tangent, self.beta, self.in_time_dim) # in Q^{p, q}

        # graph support
        if self.use_graph:
            support_h = x_q_t
            for i in range(self.graph_num_layers):
                support_h = self.graph_convs[i](support_h, edge_index) # in Q^{p, q}
                support_h = self.graph_dropouts[i](support_h)
            h_g = support_h
        
        #h_g = self.last_gcn(h_g, edge_index)

        # transformer
        h = x_q_t
        h_t = self.trans_conv(h) # in Q^{p, q}
        #h_t = self.last_gcn(h_t, edge_index)
        #h_t = self.agg(h_t, edge_index) # in Q^{p, q}
        
        # add graph embedding
        if self.use_graph:
            support = self.mid_point(h_g, h_t, self.out_time_dim , self.graph_weight) # in Q^{p, q + 1}
            if self.task == 'lp':
                return support
            support = self.manifold.extrinsic_map(support) # in Q^{p, q + 2}
            support_t = self.manifold.logmap0(support, self.beta, self.out_time_dim + 2) # in Q^{p, q + 2}

        else:
            support = h_t
            if self.task == 'lp':
                return support
            support_t = self.manifold.extrinsic_map(support) # in Q^{p, q + 1}
            support_t = self.manifold.logmap0(support_t, self.beta, self.out_time_dim + 1)
        
        # regression
        output = self.last_layer(support_t)
        return output

    def decoding_lp(self, z, edge_index):
        edge_i = edge_index[0]
        edge_j = edge_index[1]
        z_i = torch.nn.functional.embedding(edge_i, z)
        z_j = torch.nn.functional.embedding(edge_j, z)
        if self.use_graph:
            dist = self.manifold.sqdist(z_i, z_j, self.beta, self.out_time_dim + 1)
        else:
            dist = self.manifold.sqdist(z_i, z_j, self.beta, self.out_time_dim)
        return dist