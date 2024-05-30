import torch
import torch.nn as nn

def vieta_pell(n, x):
    if n == 0:
        return 2 * torch.ones_like(x)
    elif n == 1:
        return x
    else:
        return x * vieta_pell(n - 1, x) + vieta_pell(n - 2, x)

class VietaPellKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(VietaPellKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.vp_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.vp_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Normalize x to [-1, 1] using tanh
        x = torch.tanh(x)

        # Compute the Vieta-Pell basis functions
        vp_basis = []
        for n in range(self.degree + 1):
            vp_basis.append(vieta_pell(n, x))
        vp_basis = torch.stack(vp_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)

        # Compute the Vieta-Pell interpolation
        y = torch.einsum("bid,iod->bo", vp_basis, self.vp_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)

        return y