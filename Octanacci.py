import torch
import torch.nn as nn

def octanacci(n, x):
    if n == 0:
        return torch.zeros_like(x)
    elif n == 1:
        return torch.ones_like(x)
    elif n == 2 or n == 3:
        return x ** (n - 1)
    elif n == 4 or n == 5:
        return x ** (n - 2)
    elif n == 6 or n == 7:
        return x ** (n - 3)
    else:
        return x * octanacci(n-1, x) + octanacci(n-2, x) + octanacci(n-3, x) + octanacci(n-4, x) + octanacci(n-5, x) + octanacci(n-6, x) + octanacci(n-7, x) + octanacci(n-8, x)

class OctanacciKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(OctanacciKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        self.octanacci_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.octanacci_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Normalize x to [0, 1] using sigmoid
        x = torch.sigmoid(x)

        # Compute the Octanacci basis functions
        octanacci_basis = []
        for n in range(self.degree + 1):
            octanacci_basis.append(octanacci(n, x))
        octanacci_basis = torch.stack(octanacci_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)

        # Compute the Octanacci interpolation
        y = torch.einsum("bid,iod->bo", octanacci_basis, self.octanacci_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)

        return y
