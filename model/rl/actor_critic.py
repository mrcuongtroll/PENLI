"""
TODO: IMPORTANT: Since the next state in this problem is constant (i.e. given a premise, hypothesis pair, there is only
ONE explanation regardless of the model's prediction), we cannot use the curiosity approach.
Instead, we will use the ordinary actor-critic rl approach. The actor will be the model (takes the reformulated input
and outputs true, neutral, or false). And the critic will be a model that takes the concatenation of input|explanation
and output true, neutral, or false, then compare this result with the actor's. We can use the same generative model
instance as both the actor and the critic (i.e. self-criticism).
"""
from ..model import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np


class ActorCriticNetwork(nn.Module):

    def __init__(self, actor: nn.Module, critic: nn.Module):
        super(ActorCriticNetwork, self).__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, **kwargs):
        policy = self.actor(**kwargs)
        value = self.critic(**kwargs)
        return policy, value
