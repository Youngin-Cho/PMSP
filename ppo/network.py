import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical


class Network(nn.Module):
    def __init__(self, state_size, action_size, n_units):
        super(Network, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, n_units)
        self.fc2 = nn.Linear(n_units, n_units)
        self.fc3 = nn.Linear(n_units, n_units)

        self.actor = nn.Linear(n_units, action_size)
        self.critic = nn.Linear(n_units, 1)

    def act(self, state, mask, T, greedy=False):
        x = F.elu(self.fc1(state))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))

        logits = self.actor(x)
        logits = logits / T
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

        state_value = self.critic(x)

        return action.detach(), action_logprob.detach(), state_value.detach()

    def evaluate(self, batch_state, batch_action, batch_mask, T):
        x = F.elu(self.fc1(batch_state))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))

        batch_logits = self.actor(x)
        batch_logits = batch_logits / T
        batch_logits[~batch_mask] = float('-inf')

        batch_dist = Categorical(logits=batch_logits)
        batch_action_logprobs = batch_dist.log_prob(batch_action.squeeze()).unsqueeze(-1)

        batch_state_values = self.critic(x)

        batch_dist_entropys = batch_dist.entropy().unsqueeze(-1)

        return batch_action_logprobs, batch_state_values, batch_dist_entropys