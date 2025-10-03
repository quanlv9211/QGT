import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from script.manifolds.pseudohyperboloid import PseudoHyperboloid

from script.models.QGCN2.layers import PseudoLinear2, PseudoAct2

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
        self.eps = 1e-8
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
        support_x_s_norm = torch.norm(support_x_s, dim=-1, keepdim=True) + self.eps
        mask = (support_x_s_norm > self.max_norm).float()
        support_x_s_norm = torch.clamp(support_x_s_norm, max=self.max_norm)
        support_x_s_ = support_x_s / support_x_s_norm # N x (q' + 1)
        support_x_s_ = support_x_s_ * mask + support_x_s * (1 - mask)
        support_x_s_ = torch.nn.functional.normalize(support_x_s_, p=2, dim=-1)
        support_x_s_ = beta_factor * support_x_s_
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

    def __init__(self, time_dim, space_dim, beta, dropout):
        super(PseudoDropout2, self).__init__()
        self.manifold = PseudoHyperboloid(time_dim, space_dim, beta)
        self.beta = beta
        self.dropout = dropout
        self.time_dim = time_dim
        self.space_dim = space_dim
        self.eps = 1e-8
        self.max_norm = 1e6


    def forward(self, x):
        # x in Q^{p', q'}, x has shape of N x (q' + 1 + p')
        x = self.manifold.extrinsic_map(x)
        x_sh = self.manifold.q_to_sh(x, self.beta, self.time_dim + 1)  # x_sh has shape of N x (q' + 1 + p' + 1)
        x_s = x_sh[:, :self.time_dim+1] # x_s has shape of N x (q' + 1)
        x_h = x_sh[:, self.time_dim+1:] # x_h has shape of N x (p' + 1)

        support_x_s = x_s
        support_x_h = x_h[:, 1:]
        support_x_s = F.dropout(support_x_s, p=self.dropout, training=self.training) 
        support_x_h = F.dropout(support_x_h, p=self.dropout, training=self.training) 

        # transform to spherical
        beta_factor = abs(self.beta)**0.5
        support_x_s_norm = torch.norm(support_x_s, dim=-1, keepdim=True) + self.eps
        mask = (support_x_s_norm > self.max_norm).float()
        support_x_s_norm = torch.clamp(support_x_s_norm, max=self.max_norm)
        support_x_s_ = support_x_s / support_x_s_norm # N x (q' + 1)
        support_x_s_ = support_x_s_ * mask + support_x_s * (1 - mask)
        support_x_s_ = torch.nn.functional.normalize(support_x_s_, p=2, dim=-1)
        support_x_s_ = beta_factor * support_x_s_
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


class PseudoTransConvLayer(nn.Module):
    """
    Pseudo hyperboloid linear transformer layer.
    """

    def __init__(self, manifold, beta, in_channels, out_channels, num_heads, 
                use_weight=True, dropout=0.0, use_bias=False, heads_concat=True):
        # in_channels = [in_time_dim, in_space_dim]
        # out_channels = [out_time_dim, out_space_dim]
        super(PseudoTransConvLayer, self).__init__()
        self.manifold = manifold
        self.beta = beta
        self.eps = 1e-8
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
                                         in_channels, out_channels, self.beta, dropout=dropout, 
                                         use_bias=use_bias, use_hr_output=True))
            self.Wq.append(PseudoLinear2(self.manifold, self.in_time_dim, self.in_space_dim, 
                                         in_channels, out_channels, self.beta, dropout=dropout, 
                                         use_bias=use_bias, use_hr_output=True))
        if use_weight:
            for i in range(self.num_heads):
                self.Wv.append(PseudoLinear2(self.manifold, self.in_time_dim, self.in_space_dim, 
                                         in_channels, out_channels, self.beta, dropout=dropout, 
                                         use_bias=use_bias, use_hr_output=True))

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
        self.dropout_rate = args.dropout
        self.use_bn = args.trans_use_bn
        self.use_residual = args.trans_use_residual
        self.use_act = args.trans_use_act
        self.use_weight = args.trans_use_weight # Use matrix V
        self.use_bias = args.bias


        self.fcs1 = PseudoLinear2(self.manifold, self.in_time_dim, self.in_space_dim, in_channels, 
                                hidden_channels, self.beta, dropout=0.0)
        self.bns1 = PseudoLayerNorm2(self.hidden_time_dim, self.hidden_space_dim, self.beta)
        self.act  = PseudoAct2(self.manifold, self.hidden_time_dim, self.hidden_space_dim, self.beta, act)
        self.dropout1 = PseudoDropout2(self.hidden_time_dim, self.hidden_space_dim, self.beta, self.dropout_rate)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(self.num_layers):
            if self.use_residual:
                man = PseudoHyperboloid(self.hidden_time_dim, self.hidden_space_dim, self.beta)
                self.convs.append(PseudoTransConvLayer(man, self.beta, [self.hidden_time_dim, self.hidden_space_dim], 
                                    [self.hidden_time_dim, self.hidden_space_dim], self.num_heads,
                                    use_weight=self.use_weight, dropout=self.dropout_rate, use_bias=self.use_bias))
                self.bns.append(PseudoLayerNorm2(self.hidden_time_dim + 1, self.hidden_space_dim, self.beta))
                self.linears.append(PseudoLinear2(self.manifold, self.hidden_time_dim + 1, self.hidden_space_dim, 
                                        [self.hidden_time_dim + 1, self.hidden_space_dim], hidden_channels, 
                                        self.beta, dropout=0.0))
                self.dropouts.append(PseudoDropout2(self.hidden_time_dim, self.hidden_space_dim, self.beta, self.dropout_rate))
            else:
                man = PseudoHyperboloid(self.hidden_time_dim, self.hidden_space_dim, self.beta)
                self.convs.append(PseudoTransConvLayer(man, self.beta, [self.hidden_time_dim, self.hidden_space_dim], 
                                    [self.hidden_time_dim, self.hidden_space_dim], self.num_heads,
                                    use_weight=self.use_weight, dropout=self.dropout_rate, use_bias=self.use_bias))
                self.bns.append(PseudoLayerNorm2(self.hidden_time_dim , self.hidden_space_dim, self.beta))

        #man = PseudoHyperboloid(self.hidden_time_dim, self.hidden_space_dim, self.beta)
        #self.last_layer = PseudoLinear2(man, self.hidden_time_dim, self.hidden_space_dim, 
        #                                 [self.hidden_time_dim, self.hidden_space_dim], 
        #                                 out_channels, self.beta, dropout=self.dropout_rate, use_bias=self.use_bias)
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
        h = self.fcs1(x)      # Q^{p', q'}
        if self.use_bn:
            h = self.bns1(h)  # Q^{p', q'}
        if self.use_act:
            h = self.act(h)   # Q^{p', q'}
        h = self.dropout1(h)

        layer_.append(h)
        
        # linear attention
        for i in range(self.num_layers):
            h = self.convs[i](h, h)
            if self.use_residual:
                h = self.residual_connection(layer_[-1], h, self.hidden_time_dim) # in Q^{p, q + i}
            if self.use_bn: # normalize before or after? try before first
                h = self.bns[i](h)
            h = self.linears[i](h)
            #h = self.dropouts[i](h)
            layer_.append(h)

        #h = self.last_layer(h)
        #h = self.dropout2(h)
        return h

# class PseudoTransConv(nn.Module):
#     def __init__(self, manifold, beta, in_channels, hidden_channels, out_channels, act, args=None):
#         super(PseudoTransConv, self).__init__()

#         self.manifold = manifold
#         self.beta = beta

#         self.in_time_dim = in_channels[0] # q
#         self.in_space_dim = in_channels[1] # p
#         self.hidden_time_dim = hidden_channels[0] # q'
#         self.hidden_space_dim = hidden_channels[1] # p'
#         self.out_time_dim = out_channels[0] # q''
#         self.out_space_dim = out_channels[1] # p''

#         #self.num_layers = args.trans_num_layers
#         self.num_heads = args.trans_num_heads
#         self.dropout_rate = args.dropout
#         self.use_bn = args.trans_use_bn
#         self.residual = args.trans_use_residual
#         self.use_act = args.trans_use_act
#         self.use_weight = args.trans_use_weight # Use matrix V
#         self.alpha = 0.5

#         self.fcs1 = PseudoLinear2(self.manifold, self.in_time_dim, self.in_space_dim, in_channels, 
#                                       hidden_channels, self.beta, dropout=0.0)
#         self.bns1 = PseudoLayerNorm2(self.hidden_time_dim, self.hidden_space_dim, self.beta)
#         self.act  = PseudoAct2(self.manifold, self.hidden_time_dim, self.hidden_space_dim, self.beta, act)
#         self.dropout1 = PseudoDropout2(self.hidden_time_dim, self.hidden_space_dim, self.beta, self.dropout_rate)
#         # positional encoding
#         #self.add_pos_enc = args.add_positional_encoding
#         #self.positional_encoding = HypLinear(self.manifold, self.in_channels, self.hidden_channels, c_in, c_hidden)
#         #self.epsilon = torch.tensor([1.0], device=args.device)

#         #for i in range(self.num_layers):
#         self.conv1 = PseudoTransConvLayer(self.manifold, self.beta, hidden_channels, hidden_channels, self.num_heads,
#                               use_weight=self.use_weight, dropout=self.dropout_rate, use_bias=False)

#         if self.residual: 
#             self.bns2 = PseudoLayerNorm2(self.hidden_time_dim + 1, self.hidden_space_dim, self.beta)
#             self.fcs2 = PseudoLinear2(self.manifold, self.hidden_time_dim + 1, self.hidden_space_dim, [self.hidden_time_dim + 1, self.hidden_space_dim], 
#                                       out_channels, self.beta, dropout=0.0)
#             #self.dropout2 = PseudoDropout2(self.out_time_dim, self.out_space_dim, self.beta, self.dropout_rate)
#         else:
#             self.bns2 = PseudoLayerNorm2(self.hidden_time_dim, self.hidden_space_dim, self.beta)
#             self.fcs2 = PseudoLinear2(self.manifold, self.hidden_time_dim , self.hidden_space_dim, [self.hidden_time_dim, self.hidden_space_dim], 
#                                       out_channels, self.beta, dropout=0.0)
#             #self.dropout2 = PseudoDropout2(self.hidden_time_dim, self.hidden_space_dim, self.beta, self.dropout_rate)
        
#         #self.bns3 = PseudoLayerNorm2(self.out_time_dim, self.out_space_dim, self.beta)
#         self.dropout2 = PseudoDropout2(self.out_time_dim, self.out_space_dim, self.beta, self.dropout_rate)

#     def residual_connection(self, y, x, time_dim):
#         # x, y in Q^{p, q}
#         x_ex = self.manifold.extrinsic_map(x)
#         y_ex = self.manifold.extrinsic_map(y)
#         xy_ex = self.manifold.inner(x_ex, y_ex, time_dim + 1).unsqueeze(1)
#         xy_ex /= abs(self.beta)
#         origin = x_ex.clone()
#         origin[:,:] = 0
#         origin[:,0] = self.beta.abs()**0.5
#         y_ex = origin + y_ex
#         y_ex *= xy_ex
#         output = x_ex + y_ex
#         return output # output in Q^{p, q + 1}
    
#     def mid_point(self, x, y, time_dim, weight):
#         # x, y in Q^{p, q}
#         x_ex = self.manifold.extrinsic_map(x)
#         y_ex = self.manifold.extrinsic_map(y)
#         x_ex *= weight
#         y_ex *= (1 - weight)
#         z = y_ex + x_ex
#         output = self.manifold.expmap0(z, self.beta, time_dim + 1)
#         return output # output in Q^{p, q + 1}

#     def forward(self, x):
#         # the input in Q^{p, q}
#         # assume the input dimension and hidden dimension are same
#         # pre transform
#         h = self.fcs1(x)      # Q^{p', q'}
#         if self.use_bn:
#             h = self.bns1(h)  # Q^{p', q'}
#         if self.use_act:
#             h = self.act(h)   # Q^{p', q'}
#         h = self.dropout1(h)

#         # linear attention
#         z = self.conv1(h, h)   # Q^{p', q'}

#         # residual
#         if self.residual:
#             z = self.residual_connection(h, z, self.hidden_time_dim) # Q^{p', q' + 1}
#             #z = self.mid_point(h, z, self.hidden_time_dim, self.alpha) # Q^{p', q' + 1}
#         # norm
#         if self.use_bn:
#             z = self.bns2(z) # Q^{p', q' + 1}  # comment dong nay duoc 80.80
#         # ffn
#         z = self.fcs2(z)  # Q^{p', q' + 1} -> # Q^{p', q'}
#         z = self.dropout2(z)
#         return z
        

class CentralityEncoder(nn.Module):
    def __init__(self, max_degree, dim):
        """
        Centrality Encoder that assigns each node a learnable embedding
        based on its degree.

        Args:
            max_degree (int): Maximum degree of any node in the graph.
            emb_dim (int): Dimension of the centrality embedding.
        """
        super(CentralityEncoder, self).__init__()
        self.degree_embedding = nn.Embedding(max_degree + 1, dim)
        self.degree_embedding.weight.data.uniform_(-0.1, 0.1)
        #nn.init.xavier_normal_(self.degree_embedding.weight.data)

    def forward(self, degrees):
        degree_emb = self.degree_embedding(degrees)
        return degree_emb       
        
# class CentralityEncoder(nn.Module):
#     def __init__(self, max_degree, dim):
#         """
#         Centrality Encoder that assigns each node a learnable embedding
#         based on its degree.

#         Args:
#             max_degree (int): Maximum degree of any node in the graph.
#             emb_dim (int): Dimension of the centrality embedding.
#         """
#         super(CentralityEncoder, self).__init__()
#         self.degree_embedding = nn.Parameter((torch.zeros(max_degree + 1, dim)), requires_grad=False)
#         nn.init.normal_(self.degree_embedding, mean=0.0, std=1)
#         #self.degree_embedding.weight.data.uniform_(-0.1, 0.1)
#         #nn.init.xavier_normal_(self.degree_embedding.weight.data)

#     def forward(self, degrees):
#         degree_emb = self.degree_embedding[degrees]
#         return degree_emb        

   


# import math
# import torch
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.init as init
# from script.manifolds.pseudohyperboloid import PseudoHyperboloid

# from script.models.QGCN2.layers import PseudoGraphConvolution2, PseudoLinear2, PseudoAct2

# ## sua doan norm cua spherical de tranh norm = inf
# class PseudoLayerNorm2(nn.Module):
#     """
#     Pseudo hyperboloid layer normalization.
#     """

#     def __init__(self, time_dim, space_dim, beta):
#         super(PseudoLayerNorm2, self).__init__()
#         self.manifold = PseudoHyperboloid(time_dim, space_dim, beta)
#         self.beta = beta
#         self.layernorm_time = nn.LayerNorm(time_dim + 1)
#         self.layernorm_space = nn.LayerNorm(space_dim)
#         self.time_dim = time_dim
#         self.space_dim = space_dim
#         self.eps = 1e-8
#         self.max_norm = 1e6
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.layernorm_time.reset_parameters()
#         self.layernorm_space.reset_parameters()


#     def forward(self, x):
#         # x in Q^{p', q'}, x has shape of N x (q' + 1 + p')
#         x = self.manifold.extrinsic_map(x)
#         x_sh = self.manifold.q_to_sh(x, self.beta, self.time_dim + 1)  # x_sh has shape of N x (q' + 1 + p' + 1)
#         x_s = x_sh[:, :self.time_dim+1] # x_s has shape of N x (q' + 1)
#         x_h = x_sh[:, self.time_dim+1:] # x_h has shape of N x (p' + 1)

#         support_x_s = x_s
#         support_x_h = x_h[:, 1:]
#         support_x_s = self.layernorm_time(support_x_s) 
#         support_x_h = self.layernorm_space(support_x_h)

#         # transform to spherical
#         beta_factor = abs(self.beta)**0.5
#         support_x_s_norm = torch.norm(support_x_s, dim=-1, keepdim=True) + self.eps
#         mask = (support_x_s_norm > self.max_norm).float()
#         support_x_s_norm = torch.clamp(support_x_s_norm, max=self.max_norm)
#         support_x_s_ = support_x_s / support_x_s_norm # N x (q' + 1)
#         support_x_s_ = support_x_s_ * mask + support_x_s * (1 - mask)
#         support_x_s_ = torch.nn.functional.normalize(support_x_s_, p=2, dim=-1)
#         support_x_s_ = beta_factor * support_x_s_
#         assert not torch.isnan(support_x_s_).any()


#         # transform to hyperbolic
#         if self.space_dim > 0:
#             support_x_h_norm = torch.norm(support_x_h, dim=-1, keepdim=True) + self.eps
#             mask = (support_x_h_norm > self.max_norm).float()
#             support_x_h_norm = torch.clamp(support_x_h_norm, max=self.max_norm)
#             support_x_h_ = support_x_h / support_x_h_norm # N x (p')
#             support_x_h_ = torch.nn.functional.normalize(support_x_h_, p=2, dim=-1)
#             support_x_h_ = support_x_h_ * self.max_norm
#             support_x_h = support_x_h_ * mask + support_x_h * (1 - mask)
#             support_x_h_0 = (support_x_h_norm**2 - self.beta) ** 0.5  # N x 1
#             support_x_h_ = torch.cat([support_x_h_0, support_x_h], dim=-1)  # N x (p' + 1)
#         else:
#             support_x_h_ = torch.ones((x.shape[0], 1)).to(x.device)
#             support_x_h_ = support_x_h_ * beta_factor
#         assert not torch.isnan(support_x_h_).any()

#         output_sh = torch.cat((support_x_s_, support_x_h_), dim=-1) # N x (q' + 1 + p' + 1)
#         output = self.manifold.sh_to_q(output_sh, self.beta, self.time_dim) # output in Q^{p', q' + 1}
#         output = self.manifold.reverse_extrinsic_map(output) # output in Q^{p', q'}
#         return output

#     def extra_repr(self):
#         return 'beta={}'.format(
#             self.beta,
#         )

# class PseudoDropout2(nn.Module):
#     """
#     Pseudo hyperboloid layer normalization.
#     """

#     def __init__(self, time_dim, space_dim, beta, dropout):
#         super(PseudoDropout2, self).__init__()
#         self.manifold = PseudoHyperboloid(time_dim, space_dim, beta)
#         self.beta = beta
#         self.dropout = dropout
#         self.time_dim = time_dim
#         self.space_dim = space_dim
#         self.eps = 1e-8
#         self.max_norm = 1e6


#     def forward(self, x):
#         # x in Q^{p', q'}, x has shape of N x (q' + 1 + p')
#         x = self.manifold.extrinsic_map(x)
#         x_sh = self.manifold.q_to_sh(x, self.beta, self.time_dim + 1)  # x_sh has shape of N x (q' + 1 + p' + 1)
#         x_s = x_sh[:, :self.time_dim+1] # x_s has shape of N x (q' + 1)
#         x_h = x_sh[:, self.time_dim+1:] # x_h has shape of N x (p' + 1)

#         support_x_s = x_s
#         support_x_h = x_h[:, 1:]
#         support_x_s = F.dropout(support_x_s, p=self.dropout, training=self.training) 
#         support_x_h = F.dropout(support_x_h, p=self.dropout, training=self.training) 

#         # transform to spherical
#         beta_factor = abs(self.beta)**0.5
#         support_x_s_norm = torch.norm(support_x_s, dim=-1, keepdim=True) + self.eps
#         mask = (support_x_s_norm > self.max_norm).float()
#         support_x_s_norm = torch.clamp(support_x_s_norm, max=self.max_norm)
#         support_x_s_ = support_x_s / support_x_s_norm # N x (q' + 1)
#         support_x_s_ = support_x_s_ * mask + support_x_s * (1 - mask)
#         support_x_s_ = torch.nn.functional.normalize(support_x_s_, p=2, dim=-1)
#         support_x_s_ = beta_factor * support_x_s_
#         assert not torch.isnan(support_x_s_).any()


#         # transform to hyperbolic
#         if self.space_dim > 0:
#             support_x_h_norm = torch.norm(support_x_h, dim=-1, keepdim=True) + self.eps
#             mask = (support_x_h_norm > self.max_norm).float()
#             support_x_h_norm = torch.clamp(support_x_h_norm, max=self.max_norm)
#             support_x_h_ = support_x_h / support_x_h_norm # N x (p')
#             support_x_h_ = torch.nn.functional.normalize(support_x_h_, p=2, dim=-1)
#             support_x_h_ = support_x_h_ * self.max_norm
#             support_x_h = support_x_h_ * mask + support_x_h * (1 - mask)
#             support_x_h_0 = (support_x_h_norm**2 - self.beta) ** 0.5  # N x 1
#             support_x_h_ = torch.cat([support_x_h_0, support_x_h], dim=-1)  # N x (p' + 1)
#         else:
#             support_x_h_ = torch.ones((x.shape[0], 1)).to(x.device)
#             support_x_h_ = support_x_h_ * beta_factor
#         assert not torch.isnan(support_x_h_).any()

#         output_sh = torch.cat((support_x_s_, support_x_h_), dim=-1) # N x (q' + 1 + p' + 1)
#         output = self.manifold.sh_to_q(output_sh, self.beta, self.time_dim) # output in Q^{p', q' + 1}
#         output = self.manifold.reverse_extrinsic_map(output) # output in Q^{p', q'}
#         return output

#     def extra_repr(self):
#         return 'beta={}'.format(
#             self.beta,
#         )

# class PseudoGraphConv(nn.Module):
#     def __init__(self, manifold, beta, in_channels, hidden_channels, num_layers, act, dropout,
#                  use_bias=True, use_bn=False, use_residual=False, use_init=False, use_act=True):
#         super(PseudoGraphConv, self).__init__()

#         self.manifold = manifold
#         self.beta = beta
#         self.in_time_dim = in_channels[0] # q
#         self.in_space_dim = in_channels[1] # p
#         self.hidden_time_dim = hidden_channels[0] # q'
#         self.hidden_space_dim = hidden_channels[1] # p'
#         self.manifold_hid = PseudoHyperboloid(self.hidden_time_dim, self.hidden_space_dim, self.beta)
#         self.use_bn = use_bn
#         self.use_residual = use_residual
#         self.use_init = use_init
#         self.use_act = use_act
#         self.num_layers = num_layers

#         self.first_linear = PseudoLinear2(self.manifold, self.in_time_dim, self.in_space_dim, 
#                                          in_channels, hidden_channels, self.beta, dropout=dropout, use_bias=use_bias)

#         if self.use_bn:
#             self.first_bn = PseudoLayerNorm2(self.hidden_time_dim, self.hidden_space_dim, self.beta)
#         self.first_act = PseudoAct2(self.manifold_hid, self.hidden_time_dim, self.hidden_space_dim, self.beta, act)

#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()
#         for i in range(self.num_layers):
#             if self.use_residual:
#                 self.convs.append(PseudoGraphConvolution2(self.beta, [self.hidden_time_dim + i, self.hidden_space_dim], 
#                                     [self.hidden_time_dim + i, self.hidden_space_dim], dropout, act))
#                 self.bns.append(PseudoLayerNorm2(self.hidden_time_dim + i, self.hidden_space_dim, self.beta))
#             else:
#                 self.convs.append(PseudoGraphConvolution2(self.beta, [self.hidden_time_dim , self.hidden_space_dim], 
#                                     [self.hidden_time_dim , self.hidden_space_dim], dropout, act))
#                 self.bns.append(PseudoLayerNorm2(self.hidden_time_dim , self.hidden_space_dim, self.beta))


#         if self.use_residual:
#             man = PseudoHyperboloid(self.hidden_time_dim + self.num_layers, self.hidden_space_dim, self.beta)
#             self.last_layer = PseudoLinear2(man, self.hidden_time_dim + self.num_layers, self.hidden_space_dim, 
#                                          [self.hidden_time_dim + self.num_layers, self.hidden_space_dim], 
#                                          hidden_channels, self.beta, dropout=0.0, use_bias=use_bias)
#         else:
#             man = PseudoHyperboloid(self.hidden_time_dim, self.hidden_space_dim, self.beta)
#             self.last_layer = PseudoLinear2(man, self.hidden_time_dim, self.hidden_space_dim, 
#                                          [self.hidden_time_dim, self.hidden_space_dim], 
#                                          hidden_channels, self.beta, dropout=0.0, use_bias=use_bias)


#     def residual_connection(self, y, x, time_dim):
#         # x, y in Q^{p, q}
#         x_ex = self.manifold.extrinsic_map(x)
#         y_ex = self.manifold.extrinsic_map(y)
#         xy_ex = self.manifold.inner(x_ex, y_ex, time_dim + 1).unsqueeze(1)
#         xy_ex /= abs(self.beta)
#         origin = x_ex.clone()
#         origin[:,:] = 0
#         origin[:,0] = self.beta.abs()**0.5
#         y_ex = origin + y_ex
#         y_ex *= xy_ex
#         output = x_ex + y_ex
#         return output # output in Q^{p, q + 1}

#     def forward(self, x, edge_index):
#         layer_ = []

#         x = self.first_linear(x)
#         if self.use_bn:
#             x = self.first_bn(x)
#         if self.use_act:
#             x = self.first_act(x)

#         layer_.append(x)

#         for i in range(self.num_layers):
#             x = self.convs[i](x, edge_index)
#             if self.use_bn:
#                 x = self.bns[i](x)
#             if self.use_residual:
#                 x = self.residual_connection(x, layer_[-1], self.hidden_time_dim + i) # in Q^{p, q + i}
#             layer_.append(x)
#         #x = self.last_layer(x)
#         return x


# class PseudoTransConvLayer(nn.Module):
#     """
#     Pseudo hyperboloid linear transformer layer.
#     """

#     def __init__(self, manifold, beta, in_channels, out_channels, num_heads, 
#                 use_weight=True, dropout=0.0, use_bias=False, heads_concat=True):
#         # in_channels = [in_time_dim, in_space_dim]
#         # out_channels = [out_time_dim, out_space_dim]
#         super(PseudoTransConvLayer, self).__init__()
#         self.manifold = manifold
#         self.beta = beta
#         self.eps = 1e-8
#         self.max_norm = 1e6
#         self.in_time_dim = in_channels[0] # q
#         self.in_space_dim = in_channels[1] # p
#         self.out_time_dim = out_channels[0] # q'
#         self.out_space_dim = out_channels[1] # p'
#         self.num_heads = num_heads
#         self.use_weight = use_weight
#         self.heads_concat = heads_concat

#         self.Wk = nn.ModuleList()
#         self.Wq = nn.ModuleList()
#         self.Wv = nn.ModuleList()

#         for i in range(self.num_heads):
#             self.Wk.append(PseudoLinear2(self.manifold, self.in_time_dim, self.in_space_dim, 
#                                          in_channels, out_channels, self.beta, dropout=dropout, 
#                                          use_bias=use_bias, use_hr_output=True))
#             self.Wq.append(PseudoLinear2(self.manifold, self.in_time_dim, self.in_space_dim, 
#                                          in_channels, out_channels, self.beta, dropout=dropout, 
#                                          use_bias=use_bias, use_hr_output=True))
#         if use_weight:
#             for i in range(self.num_heads):
#                 self.Wv.append(PseudoLinear2(self.manifold, self.in_time_dim, self.in_space_dim, 
#                                          in_channels, out_channels, self.beta, dropout=dropout, 
#                                          use_bias=use_bias, use_hr_output=True))

#         self.final_linear_h = nn.Linear(self.num_heads * self.out_space_dim, self.out_space_dim, bias=True)
#         self.final_linear_s = nn.Linear(self.num_heads * (self.out_time_dim + 1), self.out_time_dim + 1, bias=True)
    
#     def spherical_linear_attention(self, q, k, v, output_attn=False):
#         phi_q = F.elu(q) + 1 # [N, H, q' + 1]
#         phi_k = F.elu(k) + 1 # [N, H, q' + 1]

#         # Step 1: Compute the kernel-transformed sum of K^T V across all N for each head
#         k_transpose_v = torch.einsum('nhm,nhd->hmd', phi_k, v)  # [H, q' + 1, q' + 1]

#         # Step 2: Compute the kernel-transformed dot product of Q with the above result
#         numerator = torch.einsum('nhm,hmd->nhd', phi_q, k_transpose_v)  # [N, H, q' + 1]

#         # Step 3: Compute the normalizing factor as the kernel-transformed sum of K
#         denominator = torch.einsum('nhd,hd->nh', phi_q, torch.einsum('nhd->hd', phi_k))  # [N, H]
#         denominator = denominator.unsqueeze(-1)  #

#         # Step 4: Normalize the numerator with the denominator
#         attn_output = numerator / (denominator + 1e-6)  # [N, H, q' + 1]

#         n, h, d = attn_output.shape
#         assert h == self.num_heads
#         assert d == self.out_time_dim + 1
#         if self.heads_concat:
#             attn_output = self.final_linear_s(attn_output.reshape(-1, h * d)) # [N, q' + 1]
#         else:
#             attn_output = attn_output.mean(dim=1)

#         beta_factor = abs(self.beta)**0.5
#         attn_output_norm = torch.norm(attn_output, dim=-1, keepdim=True) + self.eps # [N, 1]
#         mask = (attn_output_norm > self.max_norm).float()
#         attn_output_norm = torch.clamp(attn_output_norm, max=self.max_norm)
#         attn_output_ = attn_output / attn_output_norm # N x (q' + 1)
#         attn_output_ = attn_output_ * mask + attn_output * (1 - mask)
#         attn_output_ = torch.nn.functional.normalize(attn_output_, p=2, dim=-1)    
#         attn_output = beta_factor * attn_output_
#         assert not torch.isnan(attn_output).any()

#         return attn_output

#     def hyperbolic_linear_attention(self, q, k, v, output_attn=False):
#         qs = q[..., 1:] # [N, H, p']
#         ks = k[..., 1:] # [N, H, p']
#         vs = v[..., 1:] # [N, H, p']
#         phi_qs = F.elu(qs) + 1 # [N, H, p']
#         phi_ks = F.elu(ks) + 1 # [N, H, p']

#         # Step 1: Compute the kernel-transformed sum of K^T V across all N for each head
#         k_transpose_v = torch.einsum('nhm,nhd->hmd', phi_ks, vs)  # [H, p', p']

#         # Step 2: Compute the kernel-transformed dot product of Q with the above result
#         numerator = torch.einsum('nhm,hmd->nhd', phi_qs, k_transpose_v)  # [N, H, p']

#         # Step 3: Compute the normalizing factor as the kernel-transformed sum of K
#         denominator = torch.einsum('nhd,hd->nh', phi_qs, torch.einsum('nhd->hd', phi_ks))  # [N, H]
#         denominator = denominator.unsqueeze(-1)  #

#         # Step 4: Normalize the numerator with the denominator
#         attn_output = numerator / (denominator + 1e-6)  # [N, H, p']

#         n, h, d = attn_output.shape
#         assert h == self.num_heads
#         assert d == self.out_space_dim
#         if self.heads_concat:
#             attn_output = self.final_linear_h(attn_output.reshape(-1, h * d))
#         else:
#             attn_output = attn_output.mean(dim=1)

#         attn_output_norm = torch.norm(attn_output, dim=-1, keepdim=True) + self.eps
#         mask = (attn_output_norm > self.max_norm).float()
#         attn_output_norm = torch.clamp(attn_output_norm, max=self.max_norm)
#         attn_output_ = attn_output / attn_output_norm # N x (p')
#         attn_output_ = torch.nn.functional.normalize(attn_output_, p=2, dim=-1)
#         attn_output_ = attn_output_ * self.max_norm
#         attn_output = attn_output_ * mask + attn_output * (1 - mask)
#         attn_output_time = (attn_output_norm**2 - self.beta) ** 0.5  # N x 1
#         attn_output = torch.cat([attn_output_time, attn_output], dim=-1)  # N x p' + 1
#         assert not torch.isnan(attn_output).any()

#         return attn_output

#     def forward(self, query_input, source_input):
#         # query_input, source_input in in Q^{p, q}
#         # feature transformation
#         q_s_list = []
#         q_h_list = []
#         k_s_list = []
#         k_h_list = []
#         v_s_list = []
#         v_h_list = []
#         for i in range(self.num_heads):
#             q, time = self.Wq[i](query_input)           # list of points in Q^{p', q'}
#             q_s_list.append(q[:, :time+1].unsqueeze(1))
#             q_h_list.append(q[:, time+1:].unsqueeze(1))
#             k, time = self.Wk[i](source_input)          # list of points in Q^{p', q'}
#             k_s_list.append(k[:, :time+1].unsqueeze(1))
#             k_h_list.append(k[:, time+1:].unsqueeze(1))
#             if self.use_weight:
#                 v, time = self.Wv[i](source_input)      # list of points in Q^{p', q'}
#                 v_s_list.append(v[:, :time+1].unsqueeze(1))
#                 v_h_list.append(v[:, time+1:].unsqueeze(1))
#             else:
#                 v = self.manifold.q_to_sh(self.manifold.extrinsic_map(source_input), self.beta, self.in_time_dim + 1)
#                 v_s_list.append(v[:, :self.in_time_dim+1].unsqueeze(1))
#                 v_h_list.append(v[:, self.in_time_dim+1:].unsqueeze(1))


#         query_s = torch.cat(q_s_list, dim=1)  # [N, H, q' + 1]
#         key_s = torch.cat(k_s_list, dim=1)  # [N, H, q' + 1]
#         value_s = torch.cat(v_s_list, dim=1)  # [N, H, q' + 1]
#         query_h = torch.cat(q_h_list, dim=1)  # [N, H, p' + 1]
#         key_h = torch.cat(k_h_list, dim=1)  # [N, H, p' + 1]
#         value_h = torch.cat(v_h_list, dim=1)  # [N, H, p' + 1]

#         # linear attention on spherical
#         # if time_dim > 0:
#         att_s = self.spherical_linear_attention(query_s, key_s, value_s)

#         # linear attention on hyperbolic
#         if self.out_space_dim > 0:
#             att_h = self.hyperbolic_linear_attention(query_h, key_h, value_h)
#         else:
#             att_h = torch.ones((att_s.shape[0], 1)).to(att_s.device)
#             beta_factor = abs(self.beta)**0.5
#             att_h = att_h * beta_factor

#         # concat
#         output = torch.cat([att_s, att_h], dim=-1) # # [N, q' + 1 + p' + 1] in S x L
#         output = self.manifold.sh_to_q(output, self.beta, self.out_time_dim) # in Q^{p', q'}
#         output = self.manifold.reverse_extrinsic_map(output)
#         return output


# class PseudoTransConv(nn.Module):
#     def __init__(self, manifold, beta, in_channels, hidden_channels, out_channels, act, dropout, args=None):
#         super(PseudoTransConv, self).__init__()

#         self.manifold = manifold
#         self.beta = beta

#         self.in_time_dim = in_channels[0] # q
#         self.in_space_dim = in_channels[1] # p
#         self.hidden_time_dim = hidden_channels[0] # q'
#         self.hidden_space_dim = hidden_channels[1] # p'
#         self.manifold_hid = PseudoHyperboloid(self.hidden_time_dim, self.hidden_space_dim, self.beta)
#         self.out_time_dim = out_channels[0] # q''
#         self.out_space_dim = out_channels[1] # p''

#         self.num_layers = args.trans_num_layers
#         self.num_heads = args.trans_num_heads
#         self.dropout = dropout
#         self.use_bn = args.trans_use_bn
#         self.use_residual = args.trans_use_residual
#         self.use_act = args.trans_use_act
#         self.use_weight = args.trans_use_weight # Use matrix V
#         self.use_bias = args.bias


#         self.first_linear = PseudoLinear2(self.manifold, self.in_time_dim, self.in_space_dim, 
#                                          in_channels, hidden_channels, self.beta, dropout=self.dropout, use_bias=self.use_bias)

#         if self.use_bn:
#             self.first_bn = PseudoLayerNorm2(self.hidden_time_dim, self.hidden_space_dim, self.beta)
#         self.first_act = PseudoAct2(self.manifold_hid, self.hidden_time_dim, self.hidden_space_dim, self.beta, act)

#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()
#         for i in range(self.num_layers):
#             if self.use_residual:
#                 man = PseudoHyperboloid(self.hidden_time_dim + i, self.hidden_space_dim, self.beta)
#                 self.convs.append(PseudoTransConvLayer(man, self.beta, [self.hidden_time_dim + i, self.hidden_space_dim], 
#                                     [self.hidden_time_dim + i, self.hidden_space_dim], self.num_heads,
#                                     use_weight=self.use_weight, dropout=self.dropout, use_bias=self.use_bias))
#                 self.bns.append(PseudoLayerNorm2(self.hidden_time_dim + i, self.hidden_space_dim, self.beta))
#             else:
#                 man = PseudoHyperboloid(self.hidden_time_dim, self.hidden_space_dim, self.beta)
#                 self.convs.append(PseudoTransConvLayer(man, self.beta, [self.hidden_time_dim, self.hidden_space_dim], 
#                                     [self.hidden_time_dim, self.hidden_space_dim], self.num_heads,
#                                     use_weight=self.use_weight, dropout=self.dropout, use_bias=self.use_bias))
#                 self.bns.append(PseudoLayerNorm2(self.hidden_time_dim , self.hidden_space_dim, self.beta))

#         man = PseudoHyperboloid(self.hidden_time_dim + self.num_layers, self.hidden_space_dim, self.beta)
#         self.last_layer = PseudoLinear2(man, self.hidden_time_dim + self.num_layers, self.hidden_space_dim, 
#                                          [self.hidden_time_dim + self.num_layers, self.hidden_space_dim], 
#                                          hidden_channels, self.beta, dropout=0.0, use_bias=self.use_bias)

#     def residual_connection(self, y, x, time_dim):
#         # x, y in Q^{p, q}
#         x_ex = self.manifold.extrinsic_map(x)
#         y_ex = self.manifold.extrinsic_map(y)
#         xy_ex = self.manifold.inner(x_ex, y_ex, time_dim + 1).unsqueeze(1)
#         xy_ex /= abs(self.beta)
#         origin = x_ex.clone()
#         origin[:,:] = 0
#         origin[:,0] = self.beta.abs()**0.5
#         y_ex = origin + y_ex
#         y_ex *= xy_ex
#         output = x_ex + y_ex
#         return output # output in Q^{p, q + 1}

#     def forward(self, x):
#         layer_ = []
#         x = self.first_linear(x)      # Q^{p', q'}
#         if self.use_bn:
#             x = self.first_bn(x)  # Q^{p', q'}
#         x = self.first_act(x)   # Q^{p', q'}
#         layer_.append(x)
        
#         # linear attention
#         for i in range(self.num_layers):
#             x = self.convs[i](x, x)
#             if self.use_bn:
#                 x = self.bns[i](x)
#             if self.use_residual:
#                 x = self.residual_connection(x, layer_[-1], self.hidden_time_dim + i) # in Q^{p, q + i}
#             layer_.append(x)

#         x = self.last_layer(x)
#         return x
        

        
# class CentralityEncoder(nn.Module):
#     def __init__(self, max_degree, dim):
#         """
#         Centrality Encoder that assigns each node a learnable embedding
#         based on its degree.

#         Args:
#             max_degree (int): Maximum degree of any node in the graph.
#             emb_dim (int): Dimension of the centrality embedding.
#         """
#         super(CentralityEncoder, self).__init__()
#         self.degree_embedding = nn.Embedding(max_degree + 1, dim)
#         self.degree_embedding.weight.data.uniform_(-1, 1)
#         #nn.init.xavier_normal_(self.degree_embedding.weight.data)

#     def forward(self, degrees):
#         degree_emb = self.degree_embedding(degrees)
#         return degree_emb        

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
from script.models.QGCN2.layers import PseudoGraphConvolution2, PseudoLinear2, PseudoGraphConvolution3, PseudoAct2
from script.models.QGT.layers import PseudoLayerNorm2, PseudoTransConv, CentralityEncoder, PseudoDropout2

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
        self.dropout = args.dropout
        self.graph_num_layers = args.graph_num_layers
        self.graph_dropout = args.graph_dropout
        self.using_pretrained_feat = args.using_pretrained_feat

        self.use_pe = args.use_pe
        self.pos_enc = CentralityEncoder(args.max_node_degree, self.hidden_dim - 2)

        self.linear = nn.Linear(self.input_dim, self.hidden_dim - 2, True)
        self.gcn = GCNConv(self.input_dim, self.hidden_dim - 1, normalize=False, bias=args.bias)
        #self.pe_encoding =  nn.Linear(self.hidden_dim - 1, self.hidden_dim - 1, True)
        if self.using_pretrained_feat:
            self.feat = Parameter((torch.ones(args.num_nodes, self.input_dim)), requires_grad=True)

        self.num_layers = args.trans_num_layers

        if self.use_pe:
            self.trans_conv = PseudoTransConv(self.manifold, self.beta, self.in_features, self.out_features,
                                                self.out_features, get_activation(args.act), args)
            #self.pe_linear = PseudoLinear2(self.manifold, self.in_time_dim + 1, self.in_space_dim, [self.in_time_dim + 1, self.in_space_dim],
            #                               self.in_features, self.beta, dropout=0.0)
            #self.bn_t = PseudoLayerNorm2(self.in_time_dim, self.in_space_dim, self.beta)
            #self.a_t = PseudoAct2(self.manifold, self.in_time_dim, self.in_space_dim, self.beta, get_activation(args.act))
            #self.d_t = PseudoDropout2(self.in_time_dim, self.in_space_dim, self.beta, self.dropout)
        else:
            self.in_features = [self.in_time_dim - 1, self.in_space_dim]
            self.trans_conv = PseudoTransConv(self.manifold, self.beta, self.in_features, self.out_features,
                                                self.out_features, get_activation(args.act), args)
            man = PseudoHyperboloid(self.in_time_dim - 1, self.in_space_dim,self.beta)
            self.sub_linear = PseudoLinear2(man, self.in_time_dim - 1, self.in_space_dim, [self.in_time_dim - 1, self.in_space_dim],
                                            [self.out_time_dim, self.out_space_dim], self.beta, self.dropout)

        self.graph_convs = nn.ModuleList()
        self.in_features_g = [self.in_time_dim, self.in_space_dim]
        self.out_features_g = [self.out_time_dim, self.out_space_dim]
        for i in range(self.graph_num_layers):
            if i == 0:
                self.graph_convs.append(PseudoGraphConvolution2(self.beta, self.in_features_g, self.out_features_g, self.graph_dropout,
                                                 get_activation(args.act), use_bn=False))
            else:
                self.graph_convs.append(PseudoGraphConvolution2(self.beta, self.out_features_g, self.out_features_g, self.graph_dropout,
                                                 get_activation(args.act), use_bn=False))


        man = PseudoHyperboloid(self.out_time_dim + 1, self.out_space_dim, self.beta)
        #self.linear_t = PseudoLinear2(man, self.out_time_dim + 1, self.out_space_dim, [self.out_time_dim + 1, self.out_space_dim],
        #                                    [self.out_time_dim, self.out_space_dim], self.beta, self.dropout)
        self.bn_tt = PseudoLayerNorm2(self.out_time_dim + 1, self.out_space_dim, self.beta)
        man = PseudoHyperboloid(self.out_time_dim, self.out_space_dim, self.beta)
        #self.a_tt = PseudoAct2(man, self.out_time_dim, self.out_space_dim, self.beta, get_activation(args.act))
        #self.d_tt = PseudoDropout2(self.out_time_dim, self.out_space_dim, self.beta, self.dropout)
        #self.linear_g = PseudoLinear2(man, self.out_time_dim + 1, self.out_space_dim, [self.out_time_dim + 1, self.out_space_dim],
        #                                    [self.out_time_dim, self.out_space_dim], self.beta, self.graph_dropout)
        self.bn_g = PseudoLayerNorm2(self.out_time_dim, self.out_space_dim, self.beta)
        self.use_graph = args.use_graph
        self.graph_weight = args.graph_weight # how to use this??
        if self.use_graph:
            #self.linear_back = PseudoLinear2(man, self.out_time_dim + 1, self.out_space_dim, [self.out_time_dim + 1, self.out_space_dim],
            #                                [self.out_time_dim, self.out_space_dim], self.beta, self.graph_dropout)
            self.last_layer = nn.Linear(self.hidden_dim + 2, self.output_dim, False)
        else:
            self.last_layer = nn.Linear(self.hidden_dim + 1, self.output_dim, False)

        self.reset_parameters()
        
        # params1: trans
        # params2: gcn
        # params3: other
        self.params3 = (list(self.last_layer.parameters()))
        self.params3.extend(list(self.linear.parameters()))
        self.params3.extend(list(self.gcn.parameters()))
        #self.params3.extend(list(self.linear_back.parameters()))
        self.params3.extend(list(self.pos_enc.parameters()))
        #self.params3.extend(list(self.pe_encoding.parameters()))
        #self.params3.extend(list(self.pe_linear.parameters()))
        #self.params3.extend(list(self.bn_t.parameters()))

        self.params2 = (list(self.graph_convs.parameters()) if self.graph_convs is not None else [])
        #self.params2.extend(list(self.linear_g.parameters()))
        self.params2.extend(list(self.bn_g.parameters()))

        self.params1 = list(self.trans_conv.parameters())
        #self.params1.extend(list(self.linear_t.parameters()))
        self.params1.extend(list(self.bn_tt.parameters()))

        if self.using_pretrained_feat:
            self.params3.extend(list(self.feat))

    def reset_parameters(self):
        glorot(self.last_layer.weight)
        #self.gcn.reset_parameters()
        #self.pe_encoding.reset_parameters()

    
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

    # def add_points(self, x, y, time_dim):
    #     # x, y in Q^{p, q}
    #     x_ex = self.manifold.extrinsic_map(x)
    #     y_ex = self.manifold.extrinsic_map(y)
    #     z = y_ex + x_ex
    #     output = self.manifold.expmap0(z, self.beta, time_dim + 1)
    #     return output # output in Q^{p, q + 1}

    def forward(self, edge_index, degrees, x=None):
        if self.using_pretrained_feat:
            x = self.feat
        x_t = self.linear(x)
        x_g = self.gcn(x, edge_index.long())

        # feature map
        o = torch.zeros_like(x_t)
        x_t_tangent = torch.cat([o[:, 0:1], x_t], dim=1)
        x_q_t = self.manifold.expmap0(x_t_tangent, self.beta, self.in_time_dim - 1) # in Q^{p, q - 1}
        # feature map
        o = torch.zeros_like(x_g)
        x_g_tangent = torch.cat([o[:, 0:1], x_g], dim=1)
        x_q_g = self.manifold.expmap0(x_g_tangent, self.beta, self.in_time_dim) # in Q^{p, q}

        # graph support
        if self.use_graph:
            support_h = x_q_g
            for graph_enc in self.graph_convs:
                support_h = graph_enc(support_h, edge_index) # in Q^{p, q}
            #h_g = self.residual_connection(x_q_g, support_h, self.out_time_dim) # in Q^{p, q + 1}
            # linear
            #h_g = self.linear_g(h_g) # in Q^{p, q}
            # norm
            #h_g = self.manifold.extrinsic_map(support_h)
            h_g = self.bn_g(support_h) # need   in Q^{p, q}
            h_g = self.manifold.extrinsic_map(h_g)

        # PE
        layer_ = []
        if self.use_pe:
            #degrees = torch.clamp(degrees, max=15)
            z = self.pos_enc(degrees)
            #z = self.pe_encoding(z)
            z_t = torch.cat([o[:, 0:1], z], dim=1)
            z_q = self.manifold.expmap0(z_t, self.beta, self.in_time_dim - 1)
            h = self.mid_point(x_q_t, z_q, self.in_time_dim - 1, weight=0.5) # in Q^{p, q}
            # linear
            #h = self.pe_linear(h) # Q^{p, q}
            # norm
            #h = self.bn_t(h) # in Q^{p, q}
            # act
            #h = self.a_t(h) # in Q^{p, q}
            # dropout
            #h = self.d_t(h) # in Q^{p, q}
            layer_.append(h)
        else:
            h = x_q_t # cho linear vao day
            hh = self.sub_linear(h)
            layer_.append(hh)
        
        # transformer
        h = self.trans_conv(h) # in Q^{p, q}

        h_t = self.residual_connection(layer_[-1], h, self.out_time_dim) # in Q^{p, q + 1}
        #h_t = self.linear_t(h_t) # in Q^{p, q}
        h_t = self.bn_tt(h_t) # in Q^{p, q + 1}
        #h_t = self.a_tt(h_t)
        #h_t = self.d_tt(h_t)
        #h_t = h

        
        # add graph embedding
        if self.use_graph:
            support = self.mid_point(h_g, h_t, self.out_time_dim, self.graph_weight) # in Q^{p, q + 2}
            #support = self.linear_back(support) # in Q^{p, q}
            #support_t = self.manifold.extrinsic_map(support) # in Q^{p, q + 1}
            support_t = self.manifold.logmap0(support, self.beta, self.out_time_dim + 2) # in Q^{p, q + 2}
            #assert torch.all(support_t[:, 0] == 0) 
            #support_t = support_t[:,1:] # we can drop the first element because it is zero 

        else:
            support = h
            support_t = self.manifold.extrinsic_map(support) # in Q^{p, q + 1}
            support_t = self.manifold.logmap0(support_t, self.beta, self.out_time_dim + 1)
        
        # regression

        output = self.last_layer(support_t)
        return output


# import math
# import torch
# import numpy as np
# import torch.nn as nn
# from torch.nn import Parameter
# import torch.nn.functional as F
# import torch.nn.init as init
# from torch_geometric.nn import GCNConv

# from script.models.utils import get_activation
# from script.manifolds.pseudohyperboloid import PseudoHyperboloid
# from script.models.QGT.layers import PseudoGraphConv, PseudoTransConv, CentralityEncoder, PseudoLayerNorm2

# class QGT(nn.Module):
#     def __init__(self, args):
#         super(QGT, self).__init__()
#         self.device = args.device
#         self.input_dim = args.nfeat
#         self.in_time_dim = args.time_dim # q
#         self.in_space_dim = args.space_dim # p
#         self.out_time_dim = args.time_dim
#         self.out_space_dim = args.space_dim   # this can be changed, but I let in_ _dim == out_ _dim for stable performance
#         self.beta = nn.Parameter(torch.tensor(args.beta), requires_grad=False)
#         self.manifold = PseudoHyperboloid(self.in_time_dim, self.in_space_dim, self.beta)
#         args.nhid = args.time_dim + args.space_dim + 1
#         self.hidden_dim = args.nhid
#         self.output_dim = args.nout # nout = n_classes
#         self.in_features = [self.in_time_dim, self.in_space_dim]
#         self.out_features = [self.out_time_dim, self.out_space_dim]
#         self.using_pretrained_feat = args.using_pretrained_feat
#         self.use_pe = args.use_pe
#         self.use_graph = args.use_graph
#         #self.graph_weight = args.graph_weight # how to use this??

#         self.pos_enc = CentralityEncoder(args.max_node_degree, self.hidden_dim - 2)
#         self.linear = nn.Linear(self.input_dim, self.hidden_dim - 2, True)

#         if self.using_pretrained_feat:
#             self.feat = Parameter((torch.ones(args.num_nodes, self.input_dim)), requires_grad=True)

#         man = PseudoHyperboloid(self.in_time_dim - 1, self.in_space_dim, self.beta)
#         self.graph_enc = PseudoGraphConv(man, self.beta, [self.in_time_dim - 1, self.in_space_dim], 
#                                     [self.out_time_dim, self.out_space_dim], args.graph_num_layers, 
#                                     get_activation(args.act), args.graph_dropout)
#         #self.graph_bn = PseudoLayerNorm2(self.out_time_dim, self.out_space_dim, self.beta)

#         if self.use_pe:
#             man = PseudoHyperboloid(self.in_time_dim, self.in_space_dim, self.beta)
#             in_features = [self.in_time_dim, self.in_space_dim]
#         else:
#             man = PseudoHyperboloid(self.in_time_dim - 1, self.in_space_dim, self.beta)
#             in_features = [self.in_time_dim - 1, self.in_space_dim]

#         self.trans_enc = PseudoTransConv(man, self.beta, in_features, self.out_features, self.out_features, 
#                                             get_activation(args.act), args.dropout, args)
#         #self.trans_bn = PseudoLayerNorm2(self.out_time_dim, self.out_space_dim, self.beta)

#         if self.use_graph:
#             self.last_layer = nn.Linear(self.hidden_dim + 2, self.output_dim, False)
#         else:
#             self.last_layer = nn.Linear(self.hidden_dim + 1, self.output_dim, False)
    
#     def residual_connection(self, y, x, time_dim):
#         # x, y in Q^{p, q}
#         x_ex = self.manifold.extrinsic_map(x)
#         y_ex = self.manifold.extrinsic_map(y)
#         xy_ex = self.manifold.inner(x_ex, y_ex, time_dim + 1).unsqueeze(1)
#         xy_ex /= abs(self.beta)
#         origin = x_ex.clone()
#         origin[:,:] = 0
#         origin[:,0] = self.beta.abs()**0.5
#         y_ex = origin + y_ex
#         y_ex *= xy_ex
#         output = x_ex + y_ex
#         return output # output in Q^{p, q + 1}

#     def forward(self, edge_index, degrees, x=None):
#         if self.using_pretrained_feat:
#             x = self.feat
#         x = self.linear(x)

#         # feature map
#         o = torch.zeros_like(x)
#         x_t = torch.cat([o[:, 0:1], x], dim=1)
#         x_q = self.manifold.expmap0(x_t, self.beta, self.in_time_dim - 1) # in Q^{p, q - 1}

#         # graph support
#         if self.use_graph:
#             support_h = self.graph_enc(x_q, edge_index) # in Q^{p, q}
#             #support_h = self.graph_bn(support_h)

#         # PE
#         if self.use_pe:
#             #degrees = torch.clamp(degrees, max=10)
#             z = self.pos_enc(degrees)
#             z_t = torch.cat([o[:, 0:1], z], dim=1)
#             z_q = self.manifold.expmap0(z_t, self.beta, self.in_time_dim - 1)
#             h = self.residual_connection(x_q, z_q, self.in_time_dim - 1) # in Q^{p, q}
#         else:
#             h = x_q
        
#         # transformer
#         h = self.trans_enc(h) # in Q^{p', q'}
#         #h = self.trans_bn(h)

#         # add graph embedding
#         if self.use_graph:
#             support = self.residual_connection(h, support_h, self.out_time_dim) # in Q^{p, q + 1}
#             support_t = self.manifold.extrinsic_map(support) # in Q^{p, q + 2}
#             support_t = self.manifold.logmap0(support_t, self.beta, self.out_time_dim + 2)
#         else:
#             support = support_h
#             support_t = self.manifold.extrinsic_map(support) # in Q^{p, q + 1}
#             support_t = self.manifold.logmap0(support_t, self.beta, self.out_time_dim + 1)
        
#         # regression
#         output = self.last_layer(support_t)
#         return output
