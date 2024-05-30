import torch
import torch.nn as nn

def boas_buck(n, x, a, b):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return (a[0] * x - b[0]) / (a[1] - b[1])
    else:
        return ((a[n-1] * x - b[n-1]) * boas_buck(n-1, x, a, b) - (a[n-2] - b[n-2]) * boas_buck(n-2, x, a, b)) / (a[n] - b[n])

class BoasBuckKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(BoasBuckKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        self.a = nn.Parameter(torch.randn(degree + 1))
        self.b = nn.Parameter(torch.randn(degree + 1))

        self.boas_buck_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.boas_buck_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Normalize x to [0, 1] using sigmoid
        x = torch.sigmoid(x)

        # Compute the Boas-Buck basis functions
        boas_buck_basis = []
        for n in range(self.degree + 1):
            boas_buck_basis.append(boas_buck(n, x, self.a, self.b))
        boas_buck_basis = torch.stack(boas_buck_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)

        # Compute the Boas-Buck interpolation
        y = torch.einsum("bid,iod->bo", boas_buck_basis, self.boas_buck_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)

        return y
