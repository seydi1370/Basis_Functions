import torch
import torch.nn as nn

def heptanacci(n, x):
    if n == 0:
        return torch.zeros_like(x)
    elif n == 1:
        return torch.ones_like(x)
    elif n == 2:
        return x
    elif n == 3 or n == 4:
        return x ** (n - 2)
    elif n == 5 or n == 6:
        return x ** (n - 3)
    else:
        return x * heptanacci(n-1, x) + heptanacci(n-2, x) + heptanacci(n-3, x) + heptanacci(n-4, x) + heptanacci(n-5, x) + heptanacci(n-6, x) + heptanacci(n-7, x)

class HeptanacciKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(HeptanacciKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        self.heptanacci_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.heptanacci_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Normalize x to [0, 1] using sigmoid
        x = torch.sigmoid(x)

        # Compute the Heptanacci basis functions
        heptanacci_basis = []
        for n in range(self.degree + 1):
            heptanacci_basis.append(heptanacci(n, x))
        heptanacci_basis = torch.stack(heptanacci_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)

        # Compute the Heptanacci interpolation
        y = torch.einsum("bid,iod->bo", heptanacci_basis, self.heptanacci_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)

        return y
