import torch
import torch.nn as nn


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNZero(nn.Module):

    def __init__(self, hidden_dim, condition_dim):
        """
        hidden_dim: the dimension of the input feature
        condition_dim: the dimension of the text condition
        """
        super().__init__()
        self.nonlinear = nn.SiLU()
        self.layer = nn.Linear(condition_dim, 2 * hidden_dim, bias=True)

        nn.init.zeros_(self.layer.weight)

    def forward(self, x, condition):
        # condition: [batch_size, condition_dim], turn text condition into embedding first

        condition_embedding = self.nonlinear(condition)
        shift, scale = self.layer(condition_embedding)
        return modulate(x, shift, scale)
