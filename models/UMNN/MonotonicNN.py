import torch
import torch.nn as nn
from .NeuralIntegral import NeuralIntegral
from .ParallelNeuralIntegral import ParallelNeuralIntegral


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


class IntegrandNN(nn.Module):
    def __init__(self, in_d, hidden_layers, n_out=1):
        super(IntegrandNN, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [n_out]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net)

    def forward(self, x, h):
        return self.net(torch.cat((x, h), 1)) + 1.


class MonotonicNN(nn.Module):
    '''
    in_d : The total number of inputs
    hidden_layers : a list a the number of neurons, to be used by a network that compresses the non-monotonic variables and by the integrand net.
    nb_steps : Number of integration steps
    n_out : the number of output (each output will be monotonic w.r.t one variable)
    '''
    def __init__(self, in_d, hidden_layers, nb_steps=200, n_out=1, dev="cpu"):
        super(MonotonicNN, self).__init__()
        self.integrand = IntegrandNN(in_d, hidden_layers, n_out)
        self.net = []
        hs = [in_d-1] + hidden_layers + [2 * n_out]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        # It will output the scaling and offset factors.
        self.net = nn.Sequential(*self.net)
        self.device = dev
        self.nb_steps = nb_steps
        self.n_out = n_out

    '''
    The forward procedure takes as input x which is the variable for which the integration must be made, h are just other conditionning variables.
    '''
    def forward(self, x, h, only_derivative=False):
        x0 = torch.zeros(x.shape).to(self.device)
        out = self.net(h)
        offset = out[:, :self.n_out]
        scaling = torch.exp(out[:, self.n_out:])
        if only_derivative:
            return scaling * self.integrand(x, h)
        return scaling*ParallelNeuralIntegral.apply(x0, x, self.integrand, _flatten(self.integrand.parameters()), h, self.nb_steps) + offset

    '''
    The inverse procedure takes as input y which is the variable for which the inverse must be computed, h are just other conditionning variables.
    One output per n_out.
    y should be a scalar.
    '''
    def inverse(self, y, h, max=20, min=-20, nb_iter=20):
        idx = (torch.arange(0, self.n_out**2, self.n_out + 1).view(1, -1) + torch.arange(0, (self.n_out**2)*y.shape[0], self.n_out**2).view(-1, 1)).view(-1)
        h = h.unsqueeze(1).expand(-1, self.n_out, -1).contiguous().view(y.shape[0]*self.n_out, -1)

        # Old inversion by binary search
        x_max = torch.ones(y.shape[0], self.n_out).to(y.device) * max
        x_min = torch.ones(y.shape[0], self.n_out).to(y.device) * min
        y_max = self.forward(x_max.view(-1, 1), h).view(-1)[idx].view(-1, self.n_out)
        y_min = self.forward(x_min.view(-1, 1), h).view(-1)[idx].view(-1, self.n_out)

        for i in range(nb_iter):
            x_middle = (x_max + x_min) / 2
            y_middle = self.forward(x_middle.view(-1, 1), h).view(-1)[idx].view(-1, self.n_out)
            left = (y_middle > y).float()
            right = 1 - left
            x_max = left * x_middle + right * x_max
            x_min = right * x_middle + left * x_min
            y_max = left * y_middle + right * y_max
            y_min = right * y_middle + left * y_min
        return (x_max + x_min) / 2
