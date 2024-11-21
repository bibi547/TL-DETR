import torch
from torch import nn
from .matcher import HungarianMatcher


class Criterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.matcher = HungarianMatcher(cost_prob=100, cost_heat=1)
        self.loss_h = nn.MSELoss()
        self.loss_c = nn.CrossEntropyLoss()

    def forward(self, g_heats, probs, heats):
        indices = self.matcher(probs, heats, g_heats)

        loss = 0
        for i, idx in enumerate(indices):  # for batch_size
            gheat = g_heats[i, idx[1], :].squeeze()

            gprob = torch.zeros(probs.shape[1], dtype=torch.long).to(probs.device)
            gprob[idx[0]] = 1

            pheat = heats[i, idx[0], :]
            loss_h = self.loss_h(pheat.squeeze(), gheat.squeeze())  # 0.08
            loss_c = self.loss_c(probs.squeeze(), gprob)  # 0.65

            loss = loss + loss_c + loss_h * 10

        return loss