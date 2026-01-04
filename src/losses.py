import torch
import torch.nn as nn

class VanillaPolictGradientLoss(nn.Module):
    def __init__(self):
        super(VanillaPolictGradientLoss, self).__init__()

    def forward(self, log_probs: torch.Tensor, rewards: torch.Tensor):
        loss = -torch.sum(log_probs * rewards)
        return loss