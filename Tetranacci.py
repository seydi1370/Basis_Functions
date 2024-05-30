import torch
import torch
import torch.nn as nn

def tetranacci(n, x):
    if n == 0:
        return torch.zeros_like(x)
    elif n == 1 or n == 2 or n == 3:
        return x ** (n - 1)
    else:
        return x * tetranacci(n-1, x) + tetranacci(n-2, x) + tetranacci(n-3, x) + tetranacci(n-4, x)

class TetranacciKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(TetranacciKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        self.tetranacci_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.tetranacci_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Normalize x to [0, 1] using sigmoid
        x = torch.sigmoid(x)

        # Compute the Tetranacci basis functions
        tetranacci_basis = []
        for n in range(self.degree + 1):
            tetranacci_basis.append(tetranacci(n, x))
        tetranacci_basis = torch.stack(tetranacci_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)

        # Compute the Tetranacci interpolation
        y = torch.einsum("bid,iod->bo", tetranacci_basis, self.tetranacci_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)

        return y
