from typing import Tuple
import torch
import torch.nn as nn


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

    def forward(self, x_samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''

        :param x_samples: (b, d, h, w)
        :return:     mu : (bhw, d)
                 logvar : (bhw, d)
        '''
        x_samples = x_samples.permute(0, 2, 3, 1).contiguous()
        b, h, w, d = x_samples.shape
        flat_x = x_samples.view(-1, d)

        mu = self.p_mu(flat_x)
        logvar = self.p_logvar(flat_x)

        return mu, logvar
