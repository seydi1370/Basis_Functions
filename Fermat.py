import torch
import torch.nn as nn

def fermat(n, x):
    if n == 0:
        return torch.ones_like(x)
    else:
        return (x**(2**n) - 1) / 2

class FermatKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(FermatKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.fermat_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.fermat_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Normalize x to [0, 1] using sigmoid
        x = torch.sigmoid(x)

        # Compute the Fermat basis functions
        fermat_basis = []
        for n in range(self.degree + 1):
            fermat_basis.append(fermat(n, x))
        fermat_basis = torch.stack(fermat_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)

        # Compute the Fermat interpolation
        y = torch.einsum("bid,iod->bo", fermat_basis, self.fermat_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)

        return y