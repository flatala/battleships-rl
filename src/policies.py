import torch
import torch.nn as nn


def sample_action_get_log_prob(logits: torch.Tensor, mask: torch.Tensor):
    ''' Sample an action from the policy given logits and a mask, returning the action and its log probability. '''
    # apply mask to logits
    masked_logits = logits.masked_fill(mask == False, float('-inf'))

    # create categorical distribution and sample action
    distributions = torch.distributions.Categorical(logits=masked_logits)
    action = distributions.sample()

    # logsumexp
    log_prob = distributions.log_prob(action)

    entropy = distributions.entropy()

    return action, log_prob, entropy

class Policy_10x10(nn.Module):

    def __init__(self):
        ''' Initialize the policy network for a 10x10 Battleships board. '''
        super(Policy_10x10, self).__init__()
        self.conv1 = nn.Conv2d(2, 3, kernel_size=4) # board_size x board_size x 2 -> board_size-3 x board_size-3 x 3
        self.conv2 = nn.Conv2d(3, 1, kernel_size=3) # board_size-3 x board_size-3 x 3 -> board_size-5 x board_size-5 x 1
        self.linear1 = nn.Linear(25, 64)
        self.linear2 = nn.Linear(64, 100)

    def forward(self, state: torch.Tensor):
        ''' Forward pass through the policy network. '''
        x = self.conv1(state)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.view(-1, 25)  # flatten
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

