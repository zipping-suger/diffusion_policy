from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy

class MDNLowdimPolicy(BaseLowdimPolicy):
    def __init__(self,
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            n_gaussians=5,
            dropout=0.1
        ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_gaussians = n_gaussians

        in_channels = obs_dim * n_obs_steps
        mid_channels = 1024

        self.dense0 = nn.Linear(in_features=in_channels, out_features=mid_channels)
        self.drop0 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop2 = nn.Dropout(dropout)
        self.dense3 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop3 = nn.Dropout(dropout)

        self.pi = nn.Linear(mid_channels, n_gaussians)
        self.sigma = nn.Linear(mid_channels, n_gaussians * action_dim * n_action_steps)
        self.mu = nn.Linear(mid_channels, n_gaussians * action_dim * n_action_steps)

        self.normalizer = LinearNormalizer()

    def forward(self, obs, eps=1e-6):
        B, To, Do = obs.shape
        x = obs.reshape(B, -1)
        x = self.drop0(F.relu(self.dense0(x)))
        x = self.drop1(F.relu(self.dense1(x)))
        x = self.drop2(F.relu(self.dense2(x)))
        x = self.drop3(F.relu(self.dense3(x)))

        pi = self.pi(x).view(B, self.n_gaussians)
        sigma = self.sigma(x).view(B, self.n_gaussians, self.n_action_steps, self.action_dim)
        mu = self.mu(x).view(B, self.n_gaussians, self.n_action_steps, self.action_dim)

        log_pi = F.log_softmax(pi, dim=1)
        sigma = torch.exp(sigma + eps)

        return log_pi, sigma, mu

    def sample(self, log_pi, sigma, mu):
        B = log_pi.size(0)
        gumbel = -torch.log(-torch.log(torch.rand_like(log_pi)))
        _, indices = torch.max(log_pi + gumbel, dim=1)
        indices = indices.view(B, 1, 1, 1).expand(B, 1, self.n_action_steps, self.action_dim)
        mu = mu.gather(1, indices).squeeze(1)
        sigma = sigma.gather(1, indices).squeeze(1)
        m = torch.distributions.Normal(mu, sigma)
        action = m.sample()
        return action

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'obs' in obs_dict
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, To, Do = nobs.shape
        assert Do == self.obs_dim
        Ta = self.n_action_steps

        # only take necessary obs
        this_obs = nobs[:,:To]

        # forward pass
        log_pi, sigma, mu = self.forward(this_obs)
        action = self.sample(log_pi, sigma, mu)

        # unnormalize action
        action = self.normalizer['action'].unnormalize(action)
        result = {
            'action': action
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        nbatch = self.normalizer.normalize(batch)
        nobs = nbatch['obs']
        naction = nbatch['action']

        # shapes
        Do = self.obs_dim
        Da = self.action_dim
        To = self.n_obs_steps
        Ta = self.n_action_steps
        B = naction.shape[0]

        this_obs = nobs[:,:To]
        
        start = To - 1
        end = start + Ta
        this_action = naction[:,start:end]

        # forward pass
        log_pi, sigma, mu = self.forward(this_obs)

        # compute loss
        m = torch.distributions.Normal(mu, sigma)
        log_prob = m.log_prob(this_action.unsqueeze(1).expand_as(mu))
        log_prob = log_prob.sum(-1).sum(-1)
        loss = -torch.logsumexp(log_pi + log_prob, dim=1).mean()

        return loss