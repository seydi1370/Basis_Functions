import torch
import torch.nn as nn

def pentanacci(n, x):
    if n == 0:
        return torch.zeros_like(x)
    elif n == 1 or n == 2:
        return x ** (n - 1)
    elif n == 3 or n == 4:
        return x ** (n - 2)
    else:
        return x * pentanacci(n-1, x) + pentanacci(n-2, x) + pentanacci(n-3, x) + pentanacci(n-4, x) + pentanacci(n-5, x)

class PentanacciKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(PentanacciKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        self.pentanacci_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.pentanacci_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Normalize x to [0, 1] using sigmoid
        x = torch.sigmoid(x)

        # Compute the Pentanacci basis functions
        pentanacci_basis = []
        for n in range(self.degree + 1):
            pentanacci_basis.append(pentanacci(n, x))
        pentanacci_basis = torch.stack(pentanacci_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)

        # Compute the Pentanacci interpolation
        y = torch.einsum("bid,iod->bo", pentanacci_basis, self.pentanacci_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)

        return y

