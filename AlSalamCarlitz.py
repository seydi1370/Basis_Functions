import torch
import torch.nn as nn

def al_salam_carlitz(n, x, a, q):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return x - a
    else:
        return (x - (a + q**n)) * al_salam_carlitz(n-1, x, a, q) - a * q**(n-1) * al_salam_carlitz(n-2, x, a, q)

class AlSalamCarlitzKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a_init=None, q_init=None):
        super(AlSalamCarlitzKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        if a_init is None:
            a_init = torch.zeros(1)
        if q_init is None:
            q_init = torch.ones(1)

        self.a = nn.Parameter(a_init)
        self.q = nn.Parameter(q_init)

        self.al_salam_carlitz_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.al_salam_carlitz_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Compute the Al-Salam-Carlitz basis functions
        al_salam_carlitz_basis = []
        for n in range(self.degree + 1):
            al_salam_carlitz_basis.append(al_salam_carlitz(n, x, self.a, self.q))
        al_salam_carlitz_basis = torch.stack(al_salam_carlitz_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)
        # Compute the Al-Salam-Carlitz interpolation
        y = torch.einsum("bid,iod->bo", al_salam_carlitz_basis, self.al_salam_carlitz_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)

        return y
