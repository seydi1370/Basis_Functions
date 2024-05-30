import torch
import torch.nn as nn

def padovan(n, x):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return x
    elif n == 2:
        return x**2
    else:
        return x * padovan(n - 2, x) + padovan(n - 3, x)

class PadoKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(PadoKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.pado_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.pado_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Normalize x to [0, 1] using sigmoid
        x = torch.sigmoid(x)

        # Compute the Padovan basis functions
        pado_basis = []
        for n in range(self.degree + 1):
            pado_basis.append(padovan(n, x))
        pado_basis = torch.stack(pado_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)

        # Compute the Padovan interpolation
        y = torch.einsum("bid,iod->bo", pado_basis, self.pado_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)

        return y
