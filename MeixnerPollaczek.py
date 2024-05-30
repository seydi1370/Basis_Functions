import torch
import torch.nn as nn

def meixner_pollaczek(n, x, lambda_, phi):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return (2 * lambda_ + 1) * x + (2 * phi - 1) * lambda_
    else:
        a_n = 2 * (n + lambda_ - 1) * torch.sin(phi)
        b_n = n + 2 * lambda_ - 1
        c_n = n
        return ((a_n * x + b_n) * meixner_pollaczek(n-1, x, lambda_, phi) - c_n * meixner_pollaczek(n-2, x, lambda_, phi)) / (n + lambda_)

class MeixnerPollaczekKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(MeixnerPollaczekKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        self.lambda_ = nn.Parameter(torch.randn(1))
        self.phi = nn.Parameter(torch.randn(1))

        self.meixner_pollaczek_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.meixner_pollaczek_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Compute the Meixner-Pollaczek basis functions
        meixner_pollaczek_basis = []
        for n in range(self.degree + 1):
            meixner_pollaczek_basis.append(meixner_pollaczek(n, x, self.lambda_, self.phi))
        meixner_pollaczek_basis = torch.stack(meixner_pollaczek_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)

        # Compute the Meixner-Pollaczek interpolation
        y = torch.einsum("bid,iod->bo", meixner_pollaczek_basis, self.meixner_pollaczek_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)

        return y

