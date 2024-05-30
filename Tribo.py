import torch
import torch.nn as nn

import torch
import torch.nn as nn

def tribonacci(n, x):
    if n == 0:
        return torch.zeros_like(x)
    elif n == 1:
        return torch.ones_like(x)
    elif n == 2:
        return x
    else:
        return x * tribonacci(n - 1, x) + tribonacci(n - 2, x) + tribonacci(n - 3, x)

class TriboKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(TriboKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.tribo_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.tribo_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Normalize x to [0, 1] using sigmoid
        x = torch.sigmoid(x)

        # Compute the Tribonacci basis functions
        tribo_basis = []
        for n in range(self.degree + 1):
            tribo_basis.append(tribonacci(n, x))
        tribo_basis = torch.stack(tribo_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)

        # Compute the Tribonacci interpolation
        y = torch.einsum("bid,iod->bo", tribo_basis, self.tribo_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)

        return y
