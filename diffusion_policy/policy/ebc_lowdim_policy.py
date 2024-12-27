from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy

class EbcLowdimPolicy(BaseLowdimPolicy):
    def __init__(self,
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            dropout=0.1
        ):
        super().__init__()

        in_channels = obs_dim * n_obs_steps
        mid_channels = 1024
        out_channels = action_dim * n_action_steps

        self.dense0 = nn.Linear(in_features=in_channels, out_features=mid_channels)
        self.drop0 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop2 = nn.Dropout(dropout)
        self.dense3 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop3 = nn.Dropout(dropout)
        self.dense4 = nn.Linear(in_features=mid_channels, out_features=out_channels)

        self.normalizer = LinearNormalizer()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
    
    def forward(self, obs):
        B, To, Do = obs.shape
        x = obs.reshape(B, -1)
        x = self.drop0(torch.relu(self.dense0(x)))
        x = self.drop1(torch.relu(self.dense1(x)))
        x = self.drop2(torch.relu(self.dense2(x)))
        x = self.drop3(torch.relu(self.dense3(x)))
        x = self.dense4(x)
        x = x.reshape(B, self.n_action_steps, self.action_dim)
        return x

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
        action = self.forward(this_obs)

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
        # this_action = naction[:, :Ta]
        
        start = To - 1
        end = start + Ta
        this_action = naction[:,start:end]

        # Small additive noise to true positives.
        this_action += torch.normal(mean=0, std=1e-4,
            size=this_action.shape,
            dtype=this_action.dtype,
            device=this_action.device)

        # forward pass
        pred_action = self.forward(this_obs)

        # compute loss
        loss = F.mse_loss(pred_action, this_action)
        return loss