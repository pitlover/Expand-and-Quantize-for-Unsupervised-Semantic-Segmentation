from typing import Tuple
import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if m.bias is not None:
            m.bias.data.fill_(0)

        nn.init.xavier_uniform_(m.weight, gain=0.5)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class CLUBEncoder(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        Code from CLUB official github https://github.com/Linear95/CLUB/
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim]
    '''

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int):
        super().__init__()
        self.p_mu = nn.Sequential(nn.Linear(input_dim, hidden_dim // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim // 2, output_dim))

        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(input_dim, hidden_dim // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim // 2, output_dim),
                                      nn.Tanh())

        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''

        :param x_samples: (b, d, h, w)
        :return:     mu : (bhw, d)
                 logvar : (bhw, d)
        '''
        b, d, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (b, h, w, d)
        flat_x1 = x.view(-1, d)  # (bhw, d)

        mu = self.p_mu(flat_x1)
        logvar = self.p_logvar(flat_x1)

        return mu, logvar

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)


def Gaussian_log_likelihood(
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        reduction: str = "mean"
) -> torch.Tensor:
    b, d, h, w = x.shape
    x = x.permute(0, 2, 3, 1).contiguous()  # (b, h, w, d)
    flat_x1 = x.view(-1, d)  # (bhw, d)

    loss = -0.5 * torch.sum(
        logvar + torch.square(flat_x1 - mu) / torch.exp(logvar),
        dim=-1
    )

    if reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "mean":
        loss = torch.mean(loss)

    return loss
