import torch
import torch.nn as nn

def hexanacci(n, x):
    if n == 0:
        return torch.zeros_like(x)
    elif n == 1:
        return torch.ones_like(x)
    elif n == 2 or n == 3:
        return x ** (n - 1)
    elif n == 4 or n == 5:
        return x ** (n - 2)
    else:
        return x * hexanacci(n-1, x) + hexanacci(n-2, x) + hexanacci(n-3, x) + hexanacci(n-4, x) + hexanacci(n-5, x) + hexanacci(n-6, x)

class HexanacciKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(HexanacciKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        self.hexanacci_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.hexanacci_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Normalize x to [0, 1] using sigmoid
        x = torch.sigmoid(x)

        # Compute the Hexanacci basis functions
        hexanacci_basis = []
        for n in range(self.degree + 1):
            hexanacci_basis.append(hexanacci(n, x))
        hexanacci_basis = torch.stack(hexanacci_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)

        # Compute the Hexanacci interpolation
        y = torch.einsum("bid,iod->bo", hexanacci_basis, self.hexanacci_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)

        return y