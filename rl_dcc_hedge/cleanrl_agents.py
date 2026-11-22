# The actor-critic network (layer_init, HedgingAgent layout) in this file is adapted from
# CleanRL (https://github.com/vwxyzjn/cleanrl), used under the MIT License:
#
# MIT License
#
# Copyright (c) 2019 CleanRL developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import math

from configs.rl_config import RLExpConfig

from torch.distributions import Dirichlet
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    
class HedgingAgent(nn.Module):
    def __init__(self, args: RLExpConfig):
        super().__init__()
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim

        self.abs_action_bound = args.abs_ratio_bound

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=0.1),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.action_dim), std=0.01),
        )
        # self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))
        self.actor_logstd = nn.Parameter(torch.full((1, self.action_dim), math.log(0.3)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample() if not deterministic else probs.mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)