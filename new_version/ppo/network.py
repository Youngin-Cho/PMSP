import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical


class Network(nn.Module):
    def __init__(self, state_size, action_size, n_units):
        super(Network, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.n_units = n_units

        self.fc1 = nn.Linear(state_size, n_units)
        self.fc2 = nn.Linear(n_units, n_units)
        self.fc3 = nn.Linear(n_units, n_units)

        self.actor = nn.Linear(n_units * 2, 1)
        self.critic = nn.Linear(n_units, 1)

    def act(self, state, mask, greedy=False):
        h = F.elu(self.fc1(state))
        h = F.elu(self.fc2(h))
        h = F.elu(self.fc3(h))

        h_pooled = h.mean(dim=-2)
        h_pooled_padding = h_pooled[None, :].expand_as(h)
        h_actions = torch.cat((h, h_pooled_padding), dim=-1)

        logits = self.actor(h_actions).flatten()
        logits[~mask] = float('-inf')
        dist = Categorical(logits=logits)

        if greedy:
            action = torch.argmax(logits)
            action_logprob = dist.log_prob(action)
        else:
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            while action_logprob < -15:
                action = dist.sample()
                action_logprob = dist.log_prob(action)

        state_value = self.critic(h_pooled)

        return action.detach(), action_logprob.detach(), state_value.detach()

    def evaluate(self, batch_state, batch_action, batch_mask, batch_size):
        batch_h = F.elu(self.fc1(batch_state))
        batch_h = F.elu(self.fc2(batch_h))
        batch_h = F.elu(self.fc3(batch_h))

        batch_h = batch_h.unsqueeze(0).reshape(batch_size, -1, self.n_units)
        batch_h_pooled = batch_h.mean(dim=-2)
        batch_h_pooled_padding = batch_h_pooled[:, None, :].expand_as(batch_h)
        batch_h_actions = torch.cat((batch_h, batch_h_pooled_padding), dim=-1)

        batch_logits = self.actor(batch_h_actions).flatten(1)
        batch_logits[~batch_mask] = float('-inf')

        batch_dist = Categorical(logits=batch_logits)
        batch_action_logprobs = batch_dist.log_prob(batch_action.squeeze()).unsqueeze(-1)

        batch_state_values = self.critic(batch_h_pooled)

        batch_dist_entropys = batch_dist.entropy().unsqueeze(-1)

        return batch_action_logprobs, batch_state_values, batch_dist_entropys