"""
TODO: IMPORTANT: Since the next state in this problem is constant (i.e. given a premise, hypothesis pair, there is only
ONE explanation regardless of the model's prediction), we cannot use the curiosity approach.
Instead, we will use the ordinary actor-critic rl approach. The actor will be the model (takes the reformulated input
and outputs true, neutral, or false). And the critic will be a model that takes the concatenation of input|explanation
and output true, neutral, or false, then compare this result with the actor's. We can use the same generative model
instance as both the actor and the critic (i.e. self-criticism). The actual reward will be prediction accuracy.
"""
from ..model import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class A2C(nn.Module):

    def __init__(self, model: nn.Module, critic_hidden_size: int,
                 gamma=0.99, critic_coef=1, entropy_coef=1,
                 device='cuda'):
        """
        :param model: A model, which will act as both the actor and the critic.
        :param critic_hidden_size: The hidden dimension of the critic head.
        :param gamma: Discount parameter to estimate rewards.
        :param critic_coef: Critic loss coefficient.
        :param entropy_coef: Entropy loss coefficient.
        :param device: 'cuda' or 'cpu'.
        """
        super(A2C, self).__init__()
        self.gamma = gamma
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.model = model
        self.critic_head = nn.Sequential(
            nn.Linear(model.config.vocab_size * 2, critic_hidden_size),
            nn.ReLU(),
            nn.Linear(critic_hidden_size, 1)
        )
        self.device = device

    def forward(self,
                input_ids,
                attention_mask,
                explanation,
                explanation_mask,
                labels=None,
                **kwargs):
        # Get the policy (batch_size x output_seq_length x vocab_size)
        policy = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        # Forward the sentence and the explanation through the critic (the model acts as both the actor and the critic)
        critic_input = torch.cat([input_ids, explanation], dim=-1)
        critic_mask = torch.cat([attention_mask, explanation_mask], dim=-1)
        # (batch_size x output_seq_length x vocab_size)
        critic_logits = self.model(input_ids=critic_input,
                                   attention_mask=critic_mask,
                                   labels=labels,
                                   **kwargs)
        # Get value by forwarding the policy and the critic_logits (batch_size x output_seq_length x 2*vocab_size)
        critic_logits = torch.cat([critic_logits, policy], dim=-1)
        # value: shape = (batch_size x output_seq_length x 1)
        value = self.critic_head(critic_logits)
        return policy, value

    def train_env_episode(self, batch):
        """
        Runs one episode and collects critic values, expected return. One episode is one batch. Each state is a
        generated token.
        :return: expected reward, critic eval, action log prob, action entropy, total_reward
        """
        rewards = []

        # Run episode and save information
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        explanation = batch['explanation'].to(self.device)
        explanation_mask = batch['explanation_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Get action from actor and value from critic
        action_logits, critic_vals = self.forward(input_ids=input_ids,
                                                  attention_mask=attention_mask,
                                                  explanation=explanation,
                                                  explanation_mask=explanation_mask,
                                                  labels=labels
                                                  )
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action = action.unsqueeze(-1)
        # Get action probability. Shape = (batch_size x seq_length x 1)
        action_lp_vals = torch.gather(action_logits, 2, action)
        entropy = dist.entropy().mean()
        # Get the actual rewards. The model will be rewarded 1 for each correct prediction and 0 otherwise.
        prediction = torch.argmax(action_logits.detach(), dim=-1, keepdim=True)
        # rewards has the same shape as action_lp_vals and critic_vals
        rewards = (prediction == labels.detach().unsqueeze(-1)).float()
        # Sum the reward along the step (token positions) dimension.
        total_reward = torch.sum(rewards, dim=-2)   # TODO: keepdim or squeeze?

        # Convert reward array to expected return and standardize
        for t_i in range(rewards.shape[-2]):
            for t in range(t_i + 1, rewards.shape[-2]):
                rewards[:, t_i] += rewards[:, t] * (self.gamma ** (t - t_i))
        # Standardize rewards   # TODO: whether or not to use this
        # rewards = (rewards - torch.mean(rewards, dim=-2)) / (torch.std(rewards, dim=-2) + .000000000001)
        return rewards, critic_vals, action_lp_vals, entropy, total_reward

    def test_env_episode(self, render=True):
        """
        Run an episode of the environment in test mode
        :param render: Toggle rendering of environment :bool
        :return: Total reward :int
        """
        observation = self.env.reset()
        rewards = []
        done = False
        while not done:

            if render:
                self.env.render()

            observation = torch.from_numpy(observation).double()

            # Get action from actor
            action_logits = self.actor(observation)
            action = Categorical(logits=action_logits).sample()

            observation, reward, done, info = self.env.step(action.item())
            rewards.append(reward)

        return sum(rewards)

    def compute_loss(self, action_p_vals, G, V, critic_loss=nn.MSELoss()):
        """
        Actor Advantage Loss, where advantage = G - V
        Critic Loss, using mean squared error
        :param critic_loss: loss function for critic   :Pytorch loss module
        :param action_p_vals: Action Log Probabilities  :Tensor
        :param G: Actual Expected Returns   :Tensor
        :param V: Predicted Expected Returns    :Tensor
        :return: Actor loss tensor, Critic loss tensor  :Tensor
        """
        assert action_p_vals.shape == G.shape == V.shape
        advantage = G - V.detach()
        return -(action_p_vals * advantage.detach()).mean(), self.critic_coef*critic_loss(G, V)