import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from script.manifolds.pseudohyperboloid import PseudoHyperboloid
from torch_geometric.nn.inits import glorot, zeros
from script.models.QGCN2.layers import PseudoAct2

## sua doan norm cua spherical de tranh norm = inf
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
        beta_factor = abs(self.beta)**0.5

        # transform to spherical
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

class PseudoDropout2(nn.Module):
    """
    Pseudo hyperboloid layer normalization.
    """

    def __init__(self, time_dim, space_dim, beta, dropout_time, dropout_space):
        super(PseudoDropout2, self).__init__()
        self.manifold = PseudoHyperboloid(time_dim, space_dim, beta)
        self.beta = beta
        self.dropout_time = dropout_time
        self.dropout_space = dropout_space
        self.time_dim = time_dim
        self.space_dim = space_dim
        self.eps = 1e-5
        self.max_norm = 1e6


    def forward(self, x):
        # x in Q^{p', q'}, x has shape of N x (q' + 1 + p')
        x = self.manifold.extrinsic_map(x)
        x_sh = self.manifold.q_to_sh(x, self.beta, self.time_dim + 1)  # x_sh has shape of N x (q' + 1 + p' + 1)
        x_s = x_sh[:, :self.time_dim+1] # x_s has shape of N x (q' + 1)
        x_h = x_sh[:, self.time_dim+1:] # x_h has shape of N x (p' + 1)

        support_x_s = x_s
        support_x_h = x_h[:, 1:]
        support_x_s = F.dropout(support_x_s, p=self.dropout_time, training=self.training)  
        support_x_h = F.dropout(support_x_h, p=self.dropout_space, training=self.training) 

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

class PseudoLinear2(nn.Module):
    """
    Pseudo hyperboloid linear layer.
    """

    def __init__(self, manifold, time_dim, space_dim, in_features, out_features, beta, dropout_time=0.0, dropout_space=0.0, use_bias=True, 
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
        self.dropout_time = dropout_time
        self.dropout_space = dropout_space
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
        drop_weight1 = F.dropout(self.weight1, p=self.dropout_time, training=self.training)
        if self.out_space_dim > 0:
            drop_weight2 = F.dropout(self.weight2, p=self.dropout_space, training=self.training)
        
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



class PseudoTransConvLayer(nn.Module):
    """
    Pseudo hyperboloid linear transformer layer.
    """

    def __init__(self, manifold, beta, in_channels, out_channels, num_heads, 
                use_weight=True, dropout_time=0.0, dropout_space=0.0, use_bias=False, heads_concat=False):
        # in_channels = [in_time_dim, in_space_dim]
        # out_channels = [out_time_dim, out_space_dim]
        super(PseudoTransConvLayer, self).__init__()
        self.manifold = manifold
        self.beta = beta
        self.eps = 1e-5
        self.max_norm = 1e6
        self.in_time_dim = in_channels[0] # q
        self.in_space_dim = in_channels[1] # p
        self.out_time_dim = out_channels[0] # q'
        self.out_space_dim = out_channels[1] # p'
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.heads_concat = heads_concat

        self.Wk = nn.ModuleList()
        self.Wq = nn.ModuleList()
        self.Wv = nn.ModuleList()

        for i in range(self.num_heads):
            self.Wk.append(PseudoLinear2(self.manifold, self.in_time_dim, self.in_space_dim, 
                                         in_channels, out_channels, self.beta, dropout_time=dropout_time, dropout_space=dropout_space, 
                                         use_bias=use_bias, use_hr_output=True))
            self.Wq.append(PseudoLinear2(self.manifold, self.in_time_dim, self.in_space_dim, 
                                         in_channels, out_channels, self.beta, dropout_time=dropout_time, dropout_space=dropout_space, 
                                         use_bias=use_bias, use_hr_output=True))
        if use_weight:
            for i in range(self.num_heads):
                self.Wv.append(PseudoLinear2(self.manifold, self.in_time_dim, self.in_space_dim, 
                                         in_channels, out_channels, self.beta, dropout_time=dropout_time, dropout_space=dropout_space, 
                                         use_bias=use_bias, use_hr_output=True))
        
        if self.heads_concat:
            self.final_linear_h = nn.Linear(self.num_heads * self.out_space_dim, self.out_space_dim, bias=False)
            self.final_linear_s = nn.Linear(self.num_heads * (self.out_time_dim + 1), self.out_time_dim + 1, bias=False)
    
    def spherical_linear_attention(self, q, k, v, output_attn=False):
        phi_q = F.elu(q) + 1 # [N, H, q' + 1]
        phi_k = F.elu(k) + 1 # [N, H, q' + 1]

        # Step 1: Compute the kernel-transformed sum of K^T V across all N for each head
        k_transpose_v = torch.einsum('nhm,nhd->hmd', phi_k, v)  # [H, q' + 1, q' + 1]

        # Step 2: Compute the kernel-transformed dot product of Q with the above result
        numerator = torch.einsum('nhm,hmd->nhd', phi_q, k_transpose_v)  # [N, H, q' + 1]

        # Step 3: Compute the normalizing factor as the kernel-transformed sum of K
        denominator = torch.einsum('nhd,hd->nh', phi_q, torch.einsum('nhd->hd', phi_k))  # [N, H]
        denominator = denominator.unsqueeze(-1)  #

        # Step 4: Normalize the numerator with the denominator
        attn_output = numerator / (denominator + 1e-6)  # [N, H, q' + 1]

        n, h, d = attn_output.shape
        assert h == self.num_heads
        assert d == self.out_time_dim + 1
        if self.heads_concat:
            attn_output = self.final_linear_s(attn_output.reshape(-1, h * d)) # [N, q' + 1]
        else:
            attn_output = attn_output.mean(dim=1)

        beta_factor = abs(self.beta)**0.5
        attn_output_norm = torch.norm(attn_output, dim=-1, keepdim=True) + self.eps # [N, 1]
        mask = (attn_output_norm > self.max_norm).float()
        attn_output_norm = torch.clamp(attn_output_norm, max=self.max_norm)
        attn_output_ = attn_output / attn_output_norm # N x (q' + 1)
        attn_output_ = attn_output_ * mask + attn_output * (1 - mask)
        attn_output_ = torch.nn.functional.normalize(attn_output_, p=2, dim=-1)    
        attn_output = beta_factor * attn_output_
        assert not torch.isnan(attn_output).any()

        return attn_output

    def hyperbolic_linear_attention(self, q, k, v, output_attn=False):
        qs = q[..., 1:] # [N, H, p']
        ks = k[..., 1:] # [N, H, p']
        vs = v[..., 1:] # [N, H, p']
        phi_qs = F.elu(qs) + 1 # [N, H, p']
        phi_ks = F.elu(ks) + 1 # [N, H, p']

        # Step 1: Compute the kernel-transformed sum of K^T V across all N for each head
        k_transpose_v = torch.einsum('nhm,nhd->hmd', phi_ks, vs)  # [H, p', p']

        # Step 2: Compute the kernel-transformed dot product of Q with the above result
        numerator = torch.einsum('nhm,hmd->nhd', phi_qs, k_transpose_v)  # [N, H, p']

        # Step 3: Compute the normalizing factor as the kernel-transformed sum of K
        denominator = torch.einsum('nhd,hd->nh', phi_qs, torch.einsum('nhd->hd', phi_ks))  # [N, H]
        denominator = denominator.unsqueeze(-1)  #

        # Step 4: Normalize the numerator with the denominator
        attn_output = numerator / (denominator + 1e-8)  # [N, H, p']

        n, h, d = attn_output.shape
        assert h == self.num_heads
        assert d == self.out_space_dim
        if self.heads_concat:
            attn_output = self.final_linear_h(attn_output.reshape(-1, h * d))
        else:
            attn_output = attn_output.mean(dim=1)

        attn_output_norm = torch.norm(attn_output, dim=-1, keepdim=True) + self.eps
        mask = (attn_output_norm > self.max_norm).float()
        attn_output_norm = torch.clamp(attn_output_norm, max=self.max_norm)
        attn_output_ = attn_output / attn_output_norm # N x (p')
        attn_output_ = torch.nn.functional.normalize(attn_output_, p=2, dim=-1)
        attn_output_ = attn_output_ * self.max_norm
        attn_output = attn_output_ * mask + attn_output * (1 - mask)
        attn_output_time = (attn_output_norm**2 - self.beta) ** 0.5  # N x 1
        attn_output = torch.cat([attn_output_time, attn_output], dim=-1)  # N x p' + 1
        assert not torch.isnan(attn_output).any()

        return attn_output

    def forward(self, query_input, source_input):
        # query_input, source_input in in Q^{p, q}
        # feature transformation
        q_s_list = []
        q_h_list = []
        k_s_list = []
        k_h_list = []
        v_s_list = []
        v_h_list = []
        for i in range(self.num_heads):
            q, time = self.Wq[i](query_input)           # list of points in Q^{p', q'}
            q_s_list.append(q[:, :time+1].unsqueeze(1))
            q_h_list.append(q[:, time+1:].unsqueeze(1))
            k, time = self.Wk[i](source_input)          # list of points in Q^{p', q'}
            k_s_list.append(k[:, :time+1].unsqueeze(1))
            k_h_list.append(k[:, time+1:].unsqueeze(1))
            if self.use_weight:
                v, time = self.Wv[i](source_input)      # list of points in Q^{p', q'}
                v_s_list.append(v[:, :time+1].unsqueeze(1))
                v_h_list.append(v[:, time+1:].unsqueeze(1))
            else:
                v = self.manifold.q_to_sh(self.manifold.extrinsic_map(source_input), self.beta, self.in_time_dim + 1)
                v_s_list.append(v[:, :self.in_time_dim+1].unsqueeze(1))
                v_h_list.append(v[:, self.in_time_dim+1:].unsqueeze(1))


        query_s = torch.cat(q_s_list, dim=1)  # [N, H, q' + 1]
        key_s = torch.cat(k_s_list, dim=1)  # [N, H, q' + 1]
        value_s = torch.cat(v_s_list, dim=1)  # [N, H, q' + 1]
        query_h = torch.cat(q_h_list, dim=1)  # [N, H, p' + 1]
        key_h = torch.cat(k_h_list, dim=1)  # [N, H, p' + 1]
        value_h = torch.cat(v_h_list, dim=1)  # [N, H, p' + 1]

        # linear attention on spherical
        # if time_dim > 0:
        att_s = self.spherical_linear_attention(query_s, key_s, value_s)

        # linear attention on hyperbolic
        if self.out_space_dim > 0:
            att_h = self.hyperbolic_linear_attention(query_h, key_h, value_h)
        else:
            att_h = torch.ones((att_s.shape[0], 1)).to(att_s.device)
            beta_factor = abs(self.beta)**0.5
            att_h = att_h * beta_factor

        # concat
        output = torch.cat([att_s, att_h], dim=-1) # # [N, q' + 1 + p' + 1] in S x L
        output = self.manifold.sh_to_q(output, self.beta, self.out_time_dim) # in Q^{p', q'}
        output = self.manifold.reverse_extrinsic_map(output)
        return output

class PseudoTransConv(nn.Module):
    def __init__(self, manifold, beta, in_channels, hidden_channels, out_channels, act, args=None):
        super(PseudoTransConv, self).__init__()

        self.manifold = manifold
        self.beta = beta

        self.in_time_dim = in_channels[0] # q
        self.in_space_dim = in_channels[1] # p
        self.hidden_time_dim = hidden_channels[0] # q'
        self.hidden_space_dim = hidden_channels[1] # p'
        self.manifold_hid = PseudoHyperboloid(self.hidden_time_dim, self.hidden_space_dim, self.beta)
        self.out_time_dim = out_channels[0] # q''
        self.out_space_dim = out_channels[1] # p''

        self.num_layers = args.trans_num_layers
        self.num_heads = args.trans_num_heads
        self.dropout_time = args.dropout_time
        self.dropout_space = args.dropout_space
        self.use_bn = args.trans_use_bn
        self.use_residual = args.trans_use_residual
        self.use_act = args.trans_use_act
        self.use_weight = args.trans_use_weight # Use matrix V
        self.use_bias = args.bias
        self.alpha = args.alpha


        self.fcs1 = PseudoLinear2(self.manifold, self.in_time_dim, self.in_space_dim, in_channels, 
                                hidden_channels, self.beta, self.dropout_time, self.dropout_space)
        self.bns1 = PseudoLayerNorm2(self.hidden_time_dim, self.hidden_space_dim, self.beta)
        self.act  = PseudoAct2(self.manifold, self.hidden_time_dim, self.hidden_space_dim, self.beta, act)
        self.dropout1 = PseudoDropout2(self.hidden_time_dim, self.hidden_space_dim, self.beta, self.dropout_time, self.dropout_space)

        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(self.num_layers):
            if self.use_residual:
                man = PseudoHyperboloid(self.hidden_time_dim, self.hidden_space_dim, self.beta)
                self.convs.append(PseudoTransConvLayer(man, self.beta, [self.hidden_time_dim, self.hidden_space_dim], 
                                    [self.hidden_time_dim, self.hidden_space_dim], self.num_heads,
                                    use_weight=self.use_weight, dropout_time=self.dropout_time, dropout_space=self.dropout_space, use_bias=self.use_bias))
                self.bns.append(PseudoLayerNorm2(self.hidden_time_dim, self.hidden_space_dim, self.beta))
                self.linears.append(PseudoLinear2(self.manifold, self.hidden_time_dim + 1, self.hidden_space_dim, 
                                        [self.hidden_time_dim + 1, self.hidden_space_dim], hidden_channels, 
                                        self.beta))
                self.acts.append(PseudoAct2(self.manifold, self.hidden_time_dim, self.hidden_space_dim, self.beta, act))
                self.dropouts.append(PseudoDropout2(self.hidden_time_dim, self.hidden_space_dim, self.beta, self.dropout_time, self.dropout_space))
            else:
                man = PseudoHyperboloid(self.hidden_time_dim, self.hidden_space_dim, self.beta)
                self.convs.append(PseudoTransConvLayer(man, self.beta, [self.hidden_time_dim, self.hidden_space_dim], 
                                    [self.hidden_time_dim, self.hidden_space_dim], self.num_heads,
                                    use_weight=self.use_weight, dropout_time=self.dropout_time, dropout_space=self.dropout_space, use_bias=self.use_bias))
                self.bns.append(PseudoLayerNorm2(self.hidden_time_dim , self.hidden_space_dim, self.beta))
                self.linears.append(PseudoLinear2(self.manifold, self.hidden_time_dim, self.hidden_space_dim, 
                                        [self.hidden_time_dim, self.hidden_space_dim], hidden_channels, 
                                        self.beta))
                self.acts.append(PseudoAct2(self.manifold, self.hidden_time_dim, self.hidden_space_dim, self.beta, act))
                self.dropouts.append(PseudoDropout2(self.hidden_time_dim, self.hidden_space_dim, self.beta, self.dropout_time, self.dropout_space))
            

        man = PseudoHyperboloid(self.hidden_time_dim, self.hidden_space_dim, self.beta)
        self.last_layer = PseudoLinear2(man, self.hidden_time_dim, self.hidden_space_dim, 
                                         [self.hidden_time_dim, self.hidden_space_dim], 
                                         out_channels, self.beta, dropout_time=self.dropout_time, dropout_space=self.dropout_space, use_bias=self.use_bias)
        #self.dropout2 = PseudoDropout2(self.out_time_dim, self.out_space_dim, self.beta, self.dropout_rate)

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

    def forward(self, x):
        layer_ = []
        if self.use_bn:
            h = self.bns1(x)  # Q^{p', q'}
        else:
            h = x
        h = self.fcs1(h)      # Q^{p', q'}
        h = self.act(h)   # Q^{p', q'}
        h = self.dropout1(h)

        layer_.append(h)
        
        # linear attention
        for i in range(self.num_layers):
            if self.use_bn: # normalize before or after? try before first
                h = self.bns[i](h)
            h = self.convs[i](h, h)
            if self.use_residual:
                h = self.residual_connection(layer_[-1], h, self.hidden_time_dim) # in Q^{p, q + 1}
            h = self.linears[i](h)
            if self.use_act:
                h = self.acts[i](h)
            h = self.dropouts[i](h)
            layer_.append(h)

        h = self.last_layer(h)
        #h = self.dropout2(h)
        return h