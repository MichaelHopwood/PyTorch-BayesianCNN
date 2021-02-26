import sys
sys.path.append("..")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from metrics import calculate_kl as KL_DIV
# from ..misc import ModuleWrapper
import dgl.function as fn

# class GraphModuleWrapper(nn.Module):
#     """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

#     def __init__(self):
#         super(GraphModuleWrapper, self).__init__()

#     def set_flag(self, flag_name, value):
#         setattr(self, flag_name, value)
#         for m in self.children():
#             if hasattr(m, 'set_flag'):
#                 m.set_flag(flag_name, value)

#     def forward(self, graph, feat):
#         for module in self.children():
#             x = module(graph, feat)

#         kl = 0.0
#         for module in self.modules():
#             if hasattr(module, 'kl_loss'):
#                 kl = kl + module.kl_loss()
#         print(x,kl)
#         return x, kl

class BBBSGC(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None,
                 allow_zero_in_degree=False,
                 priors=None,
                 layer_type='lrt'):

        super(BBBSGC, self).__init__()
        # self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self.in_feats = in_feats
        self.out_feats = out_feats
        #self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self._k = k
        self._cached = cached
        self._cached_h = None
        self.use_bias = bias
        self.norm = norm
        self.layer_type = layer_type
        self._allow_zero_in_degree = allow_zero_in_degree
        self.device = torch.device("cpu")#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.empty((out_feats, in_feats), device=self.device))
        self.W_rho = Parameter(torch.empty((out_feats, in_feats), device=self.device))

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_feats), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_feats), device=self.device))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

        # nn.init.xavier_uniform_(self.fc.weight)
        # if self.fc.bias is not None:
        #     nn.init.zeros_(self.fc.bias)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, sample=True):

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise Exception('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if self._cached_h is not None:
                feat = self._cached_h
            else:
                # compute normalization
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                norm = norm.to(feat.device).unsqueeze(1)
                # compute (D^-1 A^k D)^k X
                for _ in range(self._k):
                    feat = feat * norm
                    graph.ndata['h'] = feat
                    graph.update_all(fn.message.copy_u('h', 'm'),
                                     fn.sum('m', 'h'))
                    feat = graph.ndata.pop('h')
                    feat = feat * norm

                if self.norm is not None:
                    feat = self.norm(feat)

                # cache feature
                if self._cached:
                    self._cached_h = feat
            
            if self.layer_type == 'bbb':
                # Prepare weight and bias from prior
                if self.training or sample:
                    W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
                    self.W_sigma = torch.log1p(torch.exp(self.W_rho))
                    weight = self.W_mu + W_eps * self.W_sigma

                    if self.use_bias:
                        bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                        bias = self.bias_mu + bias_eps * self.bias_sigma
                    else:
                        bias = None
                else:
                    weight = self.W_mu
                    bias = self.bias_mu if self.use_bias else None
                return F.linear(feat, weight, bias), self.kl_loss()

            elif self.layer_type == 'lrt':
                self.W_sigma = torch.log1p(torch.exp(self.W_rho))
                if self.use_bias:
                    self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                    bias_var = self.bias_sigma ** 2
                else:
                    self.bias_sigma = bias_var = None

                act_mu = F.linear(feat, self.W_mu, self.bias_mu)
                act_var = 1e-16 + F.linear(feat ** 2, self.W_sigma ** 2, bias_var)
                act_std = torch.sqrt(act_var)

                if self.training or sample:
                    eps = torch.empty(act_mu.size()).normal_(0, 1).to(self.device)
                    # print(act_mu + act_std * eps)
                    return act_mu + act_std * eps, self.kl_loss()
                else:
                    return act_mu, self.kl_loss()
                    
            else:
                raise Exception("Incorrect layer_type passed. Must be 'bbb' or 'lrt'.")

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl


"""
import math
import torch.nn as nn
from layers import BBB_Linear, BBB_Conv2d
from layers.misc import GraphModuleWrapper


class BBBLeNet(GraphModuleWrapper):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, priors, activation_type='softplus'):
        super(BBBLeNet, self).__init__()

        self.num_classes = outputs
        self.priors = priors
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")

        self.conv1 = BBBConv2d(inputs, 6, 5, padding=0, bias=True, priors=self.priors)
        self.act1 = self.act()
"""