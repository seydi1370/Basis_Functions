import torch.nn as nn

def charlier(n, x, a):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return -x / a + 1
    else:
        return ((-x / a + 2 * n - 1) * charlier(n - 1, x, a) - (n - 1) * charlier(n - 2, x, a)) / n

class CharlierKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0):
        super(CharlierKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.a = a
        self.charlier_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.charlier_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Normalize x to [0, 1] using sigmoid
        x = torch.sigmoid(x)

        # Compute the Charlier basis functions
        charlier_basis = []
        for n in range(self.degree + 1):
            charlier_basis.append(charlier(n, x, self.a))
        charlier_basis = torch.stack(charlier_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)

        # Compute the Charlier interpolation
        y = torch.einsum("bid,iod->bo", charlier_basis, self.charlier_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)

        return y