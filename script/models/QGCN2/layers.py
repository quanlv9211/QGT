import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax, add_self_loops
from torch_scatter import scatter, scatter_add
from torch_geometric.nn.conv import MessagePassing, GATConv
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot, zeros

from script.manifolds.pseudohyperboloid import PseudoHyperboloid
from script.manifolds.hyperboloid import Hyperboloid

class PseudoGraphConvolution3(nn.Module):
    """
    Pseudo hyperboloid graph convolution layer, based on hgcn
    """

    def __init__(self, beta, in_features, out_features, dropout=0.6, act=F.leaky_relu,
                 use_bias=True, use_bn=False):
        super(PseudoGraphConvolution3, self).__init__()
        # in_features = [in_time_dim, in_space_dim]
        # out_features = [out_time_dim, out_space_dim]
        self.beta = beta
        self.in_time_dim = in_features[0]
        self.in_space_dim = in_features[1]
        self.out_time_dim = out_features[0]
        self.out_space_dim = out_features[1]
        self.manifold_in = PseudoHyperboloid(self.in_time_dim, self.in_space_dim, beta)
        self.manifold_out = PseudoHyperboloid(self.out_time_dim, self.out_space_dim, beta)
        self.use_bn = use_bn
        self.agg = PseudoAgg2(self.manifold_in, self.in_time_dim, self.in_space_dim, self.beta)
        self.linear = PseudoLinear2(self.manifold_in, self.in_time_dim, self.in_space_dim, 
                                    in_features, out_features, self.beta, dropout=dropout, use_bias=use_bias)
        self.act = PseudoAct2(self.manifold_out, self.out_time_dim, self.out_space_dim, self.beta, act)
        if self.use_bn:
            self.bn = PseudoLayerNorm2(self.out_time_dim, self.out_space_dim, self.beta)

    def forward(self, x, edge_index):
        assert not torch.isnan(x).any()
        h = self.agg.forward(x, edge_index)
        assert not torch.isnan(h).any()
        h = self.linear.forward(h)
        assert not torch.isnan(h).any()
        if self.use_bn:
            h = self.bn(h)
            assert not torch.isnan(h).any()
        h = self.act.forward(h)
        assert not torch.isnan(h).any()
        return h


class PseudoGraphConvolution2(nn.Module):
    """
    Pseudo hyperboloid graph convolution layer, based on hgcn
    """

    def __init__(self, beta, in_features, out_features, dropout=0.6, act=F.leaky_relu,
                 use_bias=True, use_bn=False):
        super(PseudoGraphConvolution2, self).__init__()
        # in_features = [in_time_dim, in_space_dim]
        # out_features = [out_time_dim, out_space_dim]
        self.beta = beta
        self.in_time_dim = in_features[0]
        self.in_space_dim = in_features[1]
        self.out_time_dim = out_features[0]
        self.out_space_dim = out_features[1]
        self.manifold_in = PseudoHyperboloid(self.in_time_dim, self.in_space_dim, beta)
        self.manifold_out = PseudoHyperboloid(self.out_time_dim, self.out_space_dim, beta)
        self.use_bn = use_bn
        self.linear = PseudoLinear2(self.manifold_in, self.in_time_dim, self.in_space_dim, 
                                    in_features, out_features, self.beta, dropout=dropout, use_bias=use_bias)
        self.agg = PseudoAgg2(self.manifold_out, self.out_time_dim, self.out_space_dim, self.beta)
        self.act = PseudoAct2(self.manifold_out, self.out_time_dim, self.out_space_dim, self.beta, act)
        if self.use_bn:
            self.bn = PseudoLayerNorm2(self.out_time_dim, self.out_space_dim, self.beta)

    def forward(self, x, edge_index):
        assert not torch.isnan(x).any()
        h = self.linear.forward(x)
        assert not torch.isnan(h).any()
        h = self.agg.forward(h, edge_index)
        assert not torch.isnan(h).any()
        if self.use_bn:
            h = self.bn(h)
            assert not torch.isnan(h).any()
        h = self.act.forward(h)
        assert not torch.isnan(h).any()
        return h

class PseudoLinear2(nn.Module):
    """
    Pseudo hyperboloid linear layer.
    """

    def __init__(self, manifold, time_dim, space_dim, in_features, out_features, beta, dropout=0.6, use_bias=True, 
                 use_hr_output=False):
        # in_features = [in_time_dim, in_space_dim]
        # out_features = [out_time_dim, out_space_dim]
        super(PseudoLinear2, self).__init__()
        self.manifold = manifold
        self.time_dim = time_dim
        self.space_dim = space_dim
        self.in_time_dim = in_features[0]
        self.in_space_dim = in_features[1]
        self.out_time_dim = out_features[0]
        self.out_space_dim = out_features[1]
        self.beta = beta
        self.dropout = dropout
        self.use_bias = use_bias
        self.use_hr_output = use_hr_output
        self.bias1 = nn.Parameter(torch.Tensor(self.out_time_dim + 1), requires_grad=True)
        self.weight1 = nn.Parameter(torch.Tensor(self.out_time_dim + 1, self.in_time_dim + 1), requires_grad=True)
        if self.out_space_dim > 0:
            self.bias2 = nn.Parameter(torch.Tensor(self.out_space_dim), requires_grad=True)
            self.weight2 = nn.Parameter(torch.Tensor(self.out_space_dim, self.in_space_dim + 1), requires_grad=True)
        self.eps = 1e-5
        self.max_norm = 1e6
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        zeros(self.bias1)
        if self.out_space_dim > 0:
            glorot(self.weight2)
            zeros(self.bias2)

    def forward(self, x):
        drop_weight1 = F.dropout(self.weight1, p=self.dropout, training=self.training)
        if self.out_space_dim > 0:
            drop_weight2 = F.dropout(self.weight2, p=self.dropout, training=self.training)
        
        # x in Q^{p, q}, x has shape of N x (q + 1 + p)
        x = self.manifold.extrinsic_map(x)
        x_sh = self.manifold.q_to_sh(x, self.beta, self.time_dim + 1)  # x_sh has shape of N x (q + 1 + p + 1)
        x_s = x_sh[:, :self.time_dim+1] # x_s has shape of N x (q + 1)
        x_h = x_sh[:, self.time_dim+1:] # x_h has shape of N x (p + 1)

        mx_s = x_s @ drop_weight1.transpose(-1, -2) # mx_s has shape of N x (q' + 1)
        if self.out_space_dim > 0:
            mx_h = x_h @ drop_weight2.transpose(-1, -2) # mx_h has shape of N x (p')
        else:
            mx_h = x_h

        if self.use_bias:
            support_x_s = mx_s + self.bias1
            if self.out_space_dim > 0:
                support_x_h = mx_h + self.bias2
            else:
                support_x_h = mx_h
        else:
            support_x_s = mx_s
            support_x_h = mx_h

        # transform to spherical
        beta_factor = abs(self.beta)**0.5
        if self.time_dim > 0:
            support_x_s_norm = torch.norm(support_x_s, dim=-1, keepdim=True) + self.eps
            mask = (support_x_s_norm > self.max_norm).float()
            support_x_s_norm = torch.clamp(support_x_s_norm, max=self.max_norm)
            support_x_s_ = support_x_s / support_x_s_norm # N x (q' + 1)
            support_x_s_ = support_x_s_ * mask + support_x_s * (1 - mask)
            support_x_s_ = torch.nn.functional.normalize(support_x_s_, p=2, dim=-1)
            support_x_s_ = beta_factor * support_x_s_
        else:
            support_x_s_ = torch.ones((x.shape[0], 1)).to(x.device)
            support_x_s_ = support_x_s_ * beta_factor
        assert not torch.isnan(support_x_s_).any()

        # transform to hyperbolic
        if self.out_space_dim > 0:
            support_x_h_norm = torch.norm(support_x_h, dim=-1, keepdim=True) + self.eps
            mask = (support_x_h_norm > self.max_norm).float()
            support_x_h_norm = torch.clamp(support_x_h_norm, max=self.max_norm)
            support_x_h_ = support_x_h / support_x_h_norm # N x (p')
            support_x_h_ = torch.nn.functional.normalize(support_x_h_, p=2, dim=-1)
            support_x_h_ = support_x_h_ * self.max_norm
            support_x_h = support_x_h_ * mask + support_x_h * (1 - mask)
            support_x_h_0 = (support_x_h_norm**2 - self.beta) ** 0.5  # N x 1
            support_x_h_ = torch.cat([support_x_h_0, support_x_h], dim=-1)  # N x (p' + 1)
        else:
            support_x_h_ = torch.ones((x.shape[0], 1)).to(x.device)
            support_x_h_ = support_x_h_ * beta_factor
        assert not torch.isnan(support_x_h_).any()

        output_sh = torch.cat([support_x_s_, support_x_h_], dim=-1) # N x (q' + 1 + p' + 1)
        assert not torch.isnan(output_sh).any()
        if self.use_hr_output:
            return output_sh, self.out_time_dim
        else:
            output = self.manifold.sh_to_q(output_sh, self.beta, self.out_time_dim) # output in Q^{p', q' + 1}
            assert not torch.isnan(output).any()
            output = self.manifold.reverse_extrinsic_map(output) # output in Q^{p', q'}
            assert not torch.isnan(output).any()
            return output


    def extra_repr(self):
        return 'in_time_dim={}, in_space_dim={}, out_time_dim={}, out_space_dim={}, beta={}'.format(
            self.in_time_dim, self.in_space_dim, self.out_time_dim, self.out_space_dim, self.beta
        )


class PseudoAct2(nn.Module):
    """
    Pseudo hyperboloid activation layer.
    """

    def __init__(self, manifold, time_dim, space_dim, beta, act):
        super(PseudoAct2, self).__init__()
        self.manifold = manifold
        self.beta = beta
        self.act = act
        self.time_dim = time_dim
        self.space_dim = space_dim
        self.eps = 1e-5
        self.max_norm = 1e6

    def forward(self, x):
        # x in Q^{p', q'}, x has shape of N x (q' + 1 + p')
        x = self.manifold.extrinsic_map(x)
        assert not torch.isnan(x).any()
        x_sh = self.manifold.q_to_sh(x, self.beta, self.time_dim + 1)  # x_sh has shape of N x (q' + 1 + p' + 1)
        x_s = x_sh[:, :self.time_dim+1] # x_s has shape of N x (q' + 1)
        x_h = x_sh[:, self.time_dim+1:] # x_h has shape of N x (p' + 1)
        assert not torch.isnan(x_h).any()
        assert not torch.isnan(x_s).any()
        

        support_x_s = x_s
        support_x_h = x_h[:, 1:]
        support_x_s = self.act(support_x_s) 
        if self.space_dim > 0:
            support_x_h = self.act(support_x_h)

        # transform to spherical
        beta_factor = abs(self.beta)**0.5
        if self.time_dim > 0:
            support_x_s_norm = torch.norm(support_x_s, dim=-1, keepdim=True) + self.eps
            mask = (support_x_s_norm > self.max_norm).float()
            support_x_s_norm = torch.clamp(support_x_s_norm, max=self.max_norm)
            support_x_s_ = support_x_s / support_x_s_norm # N x (q' + 1)
            support_x_s_ = support_x_s_ * mask + support_x_s * (1 - mask)
            support_x_s_ = torch.nn.functional.normalize(support_x_s_, p=2, dim=-1)
            support_x_s_ = beta_factor * support_x_s_
        else:
            support_x_s_ = torch.ones((x.shape[0], 1)).to(x.device)
            support_x_s_ = support_x_s_ * beta_factor
        assert not torch.isnan(support_x_s_).any()
        
        # transform to hyperbolic
        if self.space_dim > 0:
            support_x_h_norm = torch.norm(support_x_h, dim=-1, keepdim=True) + self.eps
            mask = (support_x_h_norm > self.max_norm).float()
            support_x_h_norm = torch.clamp(support_x_h_norm, max=self.max_norm)
            support_x_h_ = support_x_h / support_x_h_norm # N x (p')
            support_x_h_ = torch.nn.functional.normalize(support_x_h_, p=2, dim=-1)
            support_x_h_ = support_x_h_ * self.max_norm
            support_x_h = support_x_h_ * mask + support_x_h * (1 - mask)
            support_x_h_0 = (support_x_h_norm**2 - self.beta) ** 0.5  # N x 1
            support_x_h_ = torch.cat([support_x_h_0, support_x_h], dim=-1)  # N x (p' + 1)
        else:
            support_x_h_ = torch.ones((x.shape[0], 1)).to(x.device)
            support_x_h_ = support_x_h_ * beta_factor
        assert not torch.isnan(support_x_h_).any()


        output_sh = torch.cat([support_x_s_, support_x_h_], dim=-1) # N x (q' + 1 + p' + 1)
        assert not torch.isnan(output_sh).any()
        output = self.manifold.sh_to_q(output_sh, self.beta, self.time_dim) # output in Q^{p', q' + 1}
        assert not torch.isnan(output).any()
        output = self.manifold.reverse_extrinsic_map(output) # output in Q^{p', q'}
        assert not torch.isnan(output).any()
        return output

    def extra_repr(self):
        return 'beta={}'.format(
            self.beta,
        )


class PseudoAgg2(MessagePassing):
    """
    Pseudo hyperboloid aggregation layer using degree.
    """

    def __init__(self, manifold, time_dim, space_dim, beta):
        super(PseudoAgg2, self).__init__()
        self.manifold = manifold
        self.support_manifold = Hyperboloid()
        self.time_dim = time_dim
        self.space_dim = space_dim
        self.beta = beta
        self.eps = 1e-5
        self.max_norm = 1e6

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index):
        # 
        beta_factor = abs(self.beta)**0.5
        edge_index, norm = self.norm(edge_index, x.size(0), dtype=x.dtype)
        x = self.manifold.extrinsic_map(x)
        x_sh = self.manifold.q_to_sh(x, self.beta, self.time_dim + 1)  # x_sh has shape of N x (q' + 1 + p' + 1)
        x_s = x_sh[:, :self.time_dim+1] # x_s has shape of N x (q' + 1)
        x_h = x_sh[:, self.time_dim+1:] # x_h has shape of N x (p' + 1)
        assert not torch.isnan(x_h).any()
        assert not torch.isnan(x_s).any()

        node_i = edge_index[0]
        node_j = edge_index[1]

        # aggregate on spherical
        if self.time_dim > 0:
            x_s_j = torch.nn.functional.embedding(node_j, x_s)
            support_s_j = norm.view(-1, 1) * x_s_j
            support_s_i = scatter(support_s_j, node_i, dim=0, dim_size=x.size(0))  # aggregate the neighbors of node_i
            support_s_i_norm = torch.norm(support_s_i, dim=-1, keepdim=True) + self.eps
            mask = (support_s_i_norm > self.max_norm).float()
            support_s_i_norm = torch.clamp(support_s_i_norm, max=self.max_norm)
            support_s_i_ = support_s_i / support_s_i_norm  # N x (q' + 1)
            support_s_i_ = support_s_i_ * mask + support_s_i * (1 - mask)
            support_s_i_ = torch.nn.functional.normalize(support_s_i_, p=2, dim=-1)
            support_s_i_ = support_s_i_ * beta_factor
        else:
            support_s_i_ = x_s
        assert not torch.isnan(support_s_i_).any()

        # aggregate on hyperbolic
        if self.space_dim > 0:
            x_h_j = torch.nn.functional.embedding(node_j, x_h)
            support_h_j = norm.view(-1, 1) * x_h_j
            support_h_i = scatter(support_h_j, node_i, dim=0, dim_size=x.size(0))  # aggregate the neighbors of node_i
            support_h_i_norm = torch.sqrt(torch.abs(self.support_manifold.minkowski_dot(support_h_i, support_h_i)) + self.eps) + self.eps
            mask = (support_h_i_norm > self.max_norm).float()
            support_h_i_norm = torch.clamp(support_h_i_norm, max=self.max_norm)
            support_h_i_ = support_h_i / support_h_i_norm  # N x (p' + 1)
            support_h_i_norm_ = torch.sqrt(torch.abs(self.support_manifold.minkowski_dot(support_h_i_, support_h_i_)) + self.eps) + self.eps
            support_h_i_ = support_h_i_ / support_h_i_norm_
            support_h_i_ = support_h_i_ * beta_factor
        else:
            support_h_i_ = x_h
        assert not torch.isnan(support_h_i_).any()

        output = torch.cat([support_s_i_, support_h_i_], dim=-1)
        output = self.manifold.sh_to_q(output, self.beta, self.time_dim) # output in Q^{p', q' + 1}
        assert not torch.isnan(output).any()
        output = self.manifold.reverse_extrinsic_map(output) # output in Q^{p', q'}
        assert not torch.isnan(output).any()
        return output

    def extra_repr(self):
        return 'beta={}'.format(self.beta)

class PseudoLayerNorm2(nn.Module):
    """
    Pseudo hyperboloid layer normalization.
    """

    def __init__(self, time_dim, space_dim, beta):
        super(PseudoLayerNorm2, self).__init__()
        self.manifold = PseudoHyperboloid(time_dim, space_dim, beta)
        self.beta = beta
        self.layernorm_time = nn.LayerNorm(time_dim + 1)
        self.layernorm_space = nn.LayerNorm(space_dim)
        self.time_dim = time_dim
        self.space_dim = space_dim
        self.eps = 1e-5
        self.max_norm = 1e6
        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm_time.reset_parameters()
        self.layernorm_space.reset_parameters()


    def forward(self, x):
        # x in Q^{p', q'}, x has shape of N x (q' + 1 + p')
        x = self.manifold.extrinsic_map(x)
        x_sh = self.manifold.q_to_sh(x, self.beta, self.time_dim + 1)  # x_sh has shape of N x (q' + 1 + p' + 1)
        x_s = x_sh[:, :self.time_dim+1] # x_s has shape of N x (q' + 1)
        x_h = x_sh[:, self.time_dim+1:] # x_h has shape of N x (p' + 1)

        support_x_s = x_s
        support_x_h = x_h[:, 1:]
        support_x_s = self.layernorm_time(support_x_s) 
        support_x_h = self.layernorm_space(support_x_h)

        # transform to spherical
        beta_factor = abs(self.beta)**0.5
        if self.time_dim > 0:
            support_x_s_norm = torch.norm(support_x_s, dim=-1, keepdim=True) + self.eps
            mask = (support_x_s_norm > self.max_norm).float()
            support_x_s_norm = torch.clamp(support_x_s_norm, max=self.max_norm)
            support_x_s_ = support_x_s / support_x_s_norm # N x (q' + 1)
            support_x_s_ = support_x_s_ * mask + support_x_s * (1 - mask)
            support_x_s_ = torch.nn.functional.normalize(support_x_s_, p=2, dim=-1)
            support_x_s_ = beta_factor * support_x_s_
        else:
            support_x_s_ = torch.ones((x.shape[0], 1)).to(x.device)
            support_x_s_ = support_x_s_ * beta_factor
        assert not torch.isnan(support_x_s_).any()


        # transform to hyperbolic
        if self.space_dim > 0:
            support_x_h_norm = torch.norm(support_x_h, dim=-1, keepdim=True) + self.eps
            mask = (support_x_h_norm > self.max_norm).float()
            support_x_h_norm = torch.clamp(support_x_h_norm, max=self.max_norm)
            support_x_h_ = support_x_h / support_x_h_norm # N x (p')
            support_x_h_ = torch.nn.functional.normalize(support_x_h_, p=2, dim=-1)
            support_x_h_ = support_x_h_ * self.max_norm
            support_x_h = support_x_h_ * mask + support_x_h * (1 - mask)
            support_x_h_0 = (support_x_h_norm**2 - self.beta) ** 0.5  # N x 1
            support_x_h_ = torch.cat([support_x_h_0, support_x_h], dim=-1)  # N x (p' + 1)
        else:
            support_x_h_ = torch.ones((x.shape[0], 1)).to(x.device)
            support_x_h_ = support_x_h_ * beta_factor
        assert not torch.isnan(support_x_h_).any()

        output_sh = torch.cat((support_x_s_, support_x_h_), dim=-1) # N x (q' + 1 + p' + 1)
        output = self.manifold.sh_to_q(output_sh, self.beta, self.time_dim) # output in Q^{p', q' + 1}
        output = self.manifold.reverse_extrinsic_map(output) # output in Q^{p', q'}
        return output

    def extra_repr(self):
        return 'beta={}'.format(
            self.beta,
        )