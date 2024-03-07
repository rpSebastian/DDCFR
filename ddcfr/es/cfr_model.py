from typing import Tuple

import gym
import numpy as np
import torch as th
import torch.nn as nn


class CFRNetwork(nn.Module):
    def __init__(self, input_dim: int, abg_dim: int, tau_dim: int):
        super().__init__()
        self.feature_model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
        )
        self.abg_model = nn.Sequential(
            nn.Linear(64, abg_dim),
            nn.Tanh(),
        )
        self.tau_model = nn.Sequential(nn.Linear(64, tau_dim))

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        feature = self.feature_model(x)
        abg = self.abg_model(feature)
        tau = self.tau_model(feature)
        return abg, tau

    def params(self):
        return list(self.state_dict().items())

    def load_params(self, params):
        state_dict = {k: v for k, v in params}
        self.load_state_dict(state_dict)


class CFRPolicy(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: th.device,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.build_net()

    def build_net(self):
        obs_dim = self.observation_space.shape[0]
        abg_dim = self.action_space["abg"].shape[0]
        tau_dim = self.action_space["tau"].n
        self.cfr_net = (
            CFRNetwork(obs_dim, abg_dim, tau_dim).to(th.double).to(self.device)
        )

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = self.obs_to_tensor(obs)
        with th.no_grad():
            obs = obs.reshape(1, -1)
            abg, tau_logits = self.cfr_net(obs)
            tau = th.argmax(tau_logits, dim=1, keepdim=True)
        abg = abg.cpu().numpy()[0]
        tau = tau.cpu().numpy()[0][0]
        action = dict(abg=abg, tau=tau)
        return action

    def obs_to_tensor(self, obs: np.ndarray) -> th.Tensor:
        return th.as_tensor(obs).to(self.device)

    def params(self):
        return self.cfr_net.params()

    def load_params(self, params):
        self.cfr_net.load_params(params)


class CFRModel(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: th.device,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.policy = CFRPolicy(observation_space, action_space, device)
        self.device = device

    def params(self):
        return self.policy.params()

    def load_params(self, params):
        self.policy.load_params(params)

    def forward(self, obs: np.ndarray) -> np.ndarray:
        return self.policy.predict(obs)

    def pertube(self, sigma: float = 0.05):
        eps_list = self.generate_eps()
        new_model = self.generate_new_model(eps_list, sigma)
        return new_model

    def mirror_pertube(self, sigma: float = 0.05):
        eps_list, mirror_eps_list = self.generate_eps(mirror=True)
        new_model = self.generate_new_model(eps_list, sigma)
        mirror_model = self.generate_new_model(mirror_eps_list, sigma)
        return [new_model, mirror_model]

    def generate_eps(self, zero=False, mirror=False):
        eps_list = []
        mirror_eps_list = []
        for k, v in self.params():
            eps = th.normal(0, 1, v.size())
            if zero:
                eps *= 0
            eps_list.append([k, eps])
            mirror_eps_list.append([k, -eps])
        if mirror:
            return eps_list, mirror_eps_list
        else:
            return eps_list

    def generate_new_model(self, eps_list, sigma):
        new_model = CFRModel(self.observation_space, self.action_space, self.device)
        new_model.load_params(self.params())
        new_model.set_eps_list(eps_list, sigma)
        return new_model

    def set_eps_list(self, eps_list, sigma):
        self.eps_list = eps_list
        for [(k1, v), (k2, eps)] in zip(self.params(), eps_list):
            assert k1 == k2
            v += eps * sigma

    def load(self, param_path):
        self.load_state_dict(th.load(param_path))

