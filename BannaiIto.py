
import torch
import torch.nn as nn

def bannai_ito(n, x, a, b, c):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return (x - a) / (b + c + 1)
    else:
        An = (2 * n + b + c - 1) * (2 * n + b + c) / (2 * (n + b + c))
        Cn = -(n + b - 1) * (n + c - 1) / (2 * (n + b + c))
        return ((x - An) * bannai_ito(n - 1, x, a, b, c) - Cn * bannai_ito(n - 2, x, a, b, c)) / (n + b + c)

class BannaiItoKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a_init=None, b_init=None, c_init=None):
        super(BannaiItoKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        if a_init is None:
            a_init = torch.zeros(1)
        if b_init is None:
            b_init = torch.zeros(1)
        if c_init is None:
            c_init = torch.zeros(1)

        self.a = nn.Parameter(a_init)
        self.b = nn.Parameter(b_init)
        self.c = nn.Parameter(c_init)

        self.bannai_ito_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.bannai_ito_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Normalize x to [-1, 1] using tanh
        x = torch.tanh(x)

        # Compute the Bannai-Ito basis functions
        bannai_ito_basis = []
        for n in range(self.degree + 1):
            bannai_ito_basis.append(bannai_ito(n, x, self.a, self.b, self.c))
        bannai_ito_basis = torch.stack(bannai_ito_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)

        # Compute the Bannai-Ito interpolation
        y = torch.einsum("bid,iod->bo", bannai_ito_basis, self.bannai_ito_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)

        return y