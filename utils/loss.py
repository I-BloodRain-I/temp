import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, device: torch.device, alpha=0.25, gamma=2): # gamma 2 for trend, and 1 for trading
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.to(device)

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)
