import torch
import torch.nn as nn

def askey_wilson(n, x, a, b, c, d, q):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return (2 * (1 + a * b * q) * x - (a + b) * (1 + c * d * q)) / (1 + a * b * c * d * q**2)
    else:
        An = (1 - a * b * q**(n-1)) * (1 - c * d * q**(n-1)) * (1 - a * b * c * d * q**(2*n-2))
        An /= (1 - a * b * c * d * q**(2*n-1)) * (1 - a * b * c * d * q**(2*n))
        Cn = (1 - q**n) * (1 - a * b * q**(n-1)) * (1 - c * d * q**(n-1)) * (1 - a * b * c * d * q**(2*n-2))
        Cn /= (1 - a * b * c * d * q**(2*n-2)) * (1 - a * b * c * d * q**(2*n-1))
        return ((2 * x - An) * askey_wilson(n-1, x, a, b, c, d, q) - Cn * askey_wilson(n-2, x, a, b, c, d, q)) / (1 - q**n)

class AskeyWilsonKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(AskeyWilsonKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))
        self.d = nn.Parameter(torch.randn(1))
        self.q = nn.Parameter(torch.randn(1))

        self.askey_wilson_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.askey_wilson_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        # Compute the Askey-Wilson basis functions
        askey_wilson_basis = []
        for n in range(self.degree + 1):
            askey_wilson_basis.append(askey_wilson(n, x, self.a, self.b, self.c, self.d, self.q))
        askey_wilson_basis = torch.stack(askey_wilson_basis, dim=-1)  # shape = (batch_size, input_dim, degree + 1)

        # Compute the Askey-Wilson interpolation
        y = torch.einsum("bid,iod->bo", askey_wilson_basis, self.askey_wilson_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)

        return y
