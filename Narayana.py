import torch
import torch.nn as nn

def narayana(n, x):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return x
    else:
        return x * (narayana(n-1, x) + narayana(n-2, x))

class NarayanaKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(NarayanaKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        self.narayana_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.narayana_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Normalize x to [0, 1] using sigmoid
        x = torch.sigmoid(x)

        # Compute the Narayana basis functions
        narayana_basis = []
        for n in range(self.degree + 1):
            narayana_basis.append(narayana(n, x))
        narayana_basis = torch.stack(narayana_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)

        # Compute the Narayana interpolation
        y = torch.einsum("bid,iod->bo", narayana_basis, self.narayana_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)

        return y
