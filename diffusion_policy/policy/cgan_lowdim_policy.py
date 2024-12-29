from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy

class CondGANPolicy(BaseLowdimPolicy):
    def __init__(self,
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            dropout=0.3
        ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps

        # Generator network
        in_channels = obs_dim * n_obs_steps
        mid_channels = 1024
        out_channels = action_dim * n_action_steps

        self.gen_dense0 = nn.Linear(in_features=in_channels, out_features=mid_channels)
        self.gen_dense1 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.gen_dense2 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.gen_dense3 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.gen_dense4 = nn.Linear(in_features=mid_channels, out_features=out_channels)

        # Discriminator network
        disc_in_channels = in_channels + out_channels
        self.disc_dense0 = nn.Linear(in_features=disc_in_channels, out_features=mid_channels)
        self.disc_drop0 = nn.Dropout(dropout)
        self.disc_dense1 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.disc_drop1 = nn.Dropout(dropout)
        self.disc_dense2 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.disc_drop2 = nn.Dropout(dropout)
        self.disc_dense3 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.disc_drop3 = nn.Dropout(dropout)
        self.disc_dense4 = nn.Linear(in_features=mid_channels, out_features=1)

        self.normalizer = LinearNormalizer()

    def forward(self, obs):
        B, To, Do = obs.shape
        x = obs.reshape(B, -1)
        x = F.leaky_relu(self.gen_dense0(x), 0.2, inplace=True)
        x = F.leaky_relu(self.gen_dense1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.gen_dense2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.gen_dense3(x), 0.2, inplace=True)
        x = self.gen_dense4(x)
        x = x.reshape(B, self.n_action_steps, self.action_dim)
        return x

    def discriminator_forward(self, obs, action):
        B, To, Do = obs.shape
        B, Ta, Da = action.shape
        obs = obs.reshape(B, -1)
        action = action.reshape(B, -1)
        x = torch.cat([obs, action], dim=-1)
        x = self.disc_drop0(F.leaky_relu(self.disc_dense0(x), 0.2, inplace=True))
        x = self.disc_drop1(F.leaky_relu(self.disc_dense1(x), 0.2, inplace=True))
        x = self.disc_drop2(F.leaky_relu(self.disc_dense2(x), 0.2, inplace=True))
        x = self.disc_drop3(F.leaky_relu(self.disc_dense3(x), 0.2, inplace=True))
        x = self.disc_dense4(x)
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

        # forward pass for generator
        pred_action = self.forward(this_obs)

        # discriminator loss
        real_labels = torch.ones(B, 1).to(this_obs.device)
        fake_labels = torch.zeros(B, 1).to(this_obs.device)

        real_output = self.discriminator_forward(this_obs, this_action)
        fake_output = self.discriminator_forward(this_obs, pred_action.detach())

        d_loss_real = F.binary_cross_entropy_with_logits(real_output, real_labels)
        d_loss_fake = F.binary_cross_entropy_with_logits(fake_output, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        # generator loss
        fake_output = self.discriminator_forward(this_obs, pred_action)
        g_loss = F.binary_cross_entropy_with_logits(fake_output, real_labels)

        return g_loss, d_loss