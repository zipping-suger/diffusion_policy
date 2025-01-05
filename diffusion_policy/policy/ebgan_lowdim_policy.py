from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy

class EbganLowdimPolicy(BaseLowdimPolicy):
    def __init__(self,
                 horizon,
                 obs_dim,
                 action_dim,
                 n_action_steps,
                 n_obs_steps,
                 dropout=0.1,
                 pred_n_samples=16384,
                 train_n_neg=128):
        super().__init__()
        
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.pred_n_samples = pred_n_samples
        self.train_n_neg = train_n_neg

        # Generator network
        in_channels = obs_dim * n_obs_steps
        mid_channels = 1024
        out_channels = action_dim * n_action_steps

        self.gen_dense0 = nn.Linear(in_features=in_channels, out_features=mid_channels)
        self.gen_dense1 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.gen_dense2 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.gen_dense3 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.gen_dense4 = nn.Linear(in_features=mid_channels, out_features=out_channels)

        # Energy network
        ebm_in_channels = in_channels + out_channels
        self.ebm_dense0 = nn.Linear(in_features=ebm_in_channels, out_features=mid_channels)
        self.ebm_drop0 = nn.Dropout(dropout)
        self.ebm_dense1 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.ebm_drop1 = nn.Dropout(dropout)
        self.ebm_dense2 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.ebm_drop2 = nn.Dropout(dropout)
        self.ebm_dense3 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.ebm_drop3 = nn.Dropout(dropout)
        self.ebm_dense4 = nn.Linear(in_features=mid_channels, out_features=1)

        self.normalizer = LinearNormalizer()

    def forward(self, obs):
        B, To, Do = obs.shape
        x = obs.reshape(B, -1)
        x = torch.relu(self.gen_dense0(x))
        x = torch.relu(self.gen_dense1(x))
        x = torch.relu(self.gen_dense2(x))
        x = torch.relu(self.gen_dense3(x))
        x = self.gen_dense4(x)
        x = x.reshape(B, self.n_action_steps, self.action_dim)
        return x

    def forward_energy_model(self, obs, action):
        B, N, Ta, Da = action.shape
        B, To, Do = obs.shape
        s = obs.reshape(B,1,-1).expand(-1,N,-1)
        x = torch.cat([s, action.reshape(B,N,-1)], dim=-1).reshape(B*N,-1)
        x = self.ebm_drop0(torch.relu(self.ebm_dense0(x)))
        x = self.ebm_drop1(torch.relu(self.ebm_dense1(x)))
        x = self.ebm_drop2(torch.relu(self.ebm_dense2(x)))
        x = self.ebm_drop3(torch.relu(self.ebm_dense3(x)))
        x = self.ebm_dense4(x)
        x = x.reshape(B,N)
        return x
    
    
    def compute_loss(self, batch):
        # Normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        nobs = nbatch['obs']
        naction = nbatch['action']

        # Shapes
        To = self.n_obs_steps
        Ta = self.n_action_steps
        B = naction.shape[0]

        this_obs = nobs[:, :To]
        start = To - 1
        end = start + Ta
        this_action = naction[:, start:end]

        # Small additive noise to true positives
        this_action = this_action + torch.normal(mean=0, std=1e-4,
                                                size=this_action.shape,
                                                dtype=this_action.dtype,
                                                device=this_action.device)

        # Generate actions
        with torch.no_grad():
            generated_action = self.forward(this_obs)
            generated_action = generated_action.unsqueeze(1)

        # Sample negatives: (B, train_n_neg, Ta, Da)
        naction_stats = self.get_naction_stats()
        action_dist = torch.distributions.Uniform(
            low=naction_stats['min'],
            high=naction_stats['max']
        )
        
        negative_samples = action_dist.sample((B, self.train_n_neg, Ta)).to(
            dtype=this_action.dtype)
        
        counter_samples = negative_samples

        # counter_samples = generated_action
        
        # InfoNCE-style loss function
        positive_energy = -self.forward_energy_model(this_obs, this_action.unsqueeze(1))
        neg_energies = -self.forward_energy_model(this_obs, counter_samples)
        denominator = torch.logsumexp(torch.cat([positive_energy, neg_energies], dim=-1), dim=-1)
        e_loss =  torch.mean(denominator - positive_energy)
    
            
        # Compute generator loss, minimize energy
        g_action = self.forward(this_obs)
        # Make it (B, 1, Ta, Da)
        g_action = g_action.unsqueeze(1)
        g_energy = self.forward_energy_model(this_obs, g_action)
        g_loss = torch.mean(g_energy)
        
        return e_loss, g_loss
        
    def get_naction_stats(self):
        Da = self.action_dim
        naction_stats = self.normalizer['action'].get_output_stats()
        repeated_stats = dict()
        for key, value in naction_stats.items():
            assert len(value.shape) == 1
            n_repeats = Da // value.shape[0]
            assert value.shape[0] * n_repeats == Da
            repeated_stats[key] = value.repeat(n_repeats)
        return repeated_stats

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'obs' in obs_dict
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim

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
