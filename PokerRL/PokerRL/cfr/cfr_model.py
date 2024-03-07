from typing import Dict, List, Tuple

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
    ):
        super().__init__()
        self.build_net()

    def build_net(self):
        obs_dim = 2
        abg_dim = 3
        tau_dim = 5
        self.cfr_net = CFRNetwork(obs_dim, abg_dim, tau_dim).to(th.double)

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
        return th.as_tensor(obs)

    def params(self):
        return self.cfr_net.params()

    def load_params(self, params):
        self.cfr_net.load_params(params)


class CFRModel(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.policy = CFRPolicy()

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


class A2CPolicy(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.build_net()

    def build_net(self):
        self.ac_net = self.make_ac_net()

    def make_ac_net(self) -> "ACNetwork":
        obs_dim = 2
        abg_dim = 3
        tau_dim = 5
        self.abg_log_std = nn.Parameter(th.zeros(abg_dim), requires_grad=True)
        return ACNetwork(obs_dim, abg_dim, tau_dim).to(th.double)

    def predict(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = self.obs_to_tensor(obs)
        with th.no_grad():
            abg_logits, tau_logits, value = self.ac_net(obs)

            abg_std = self.abg_log_std.exp()
            abg_pi = th.distributions.Normal(abg_logits, abg_std)
            abg = abg_pi.sample()
            abg_log_prob = abg_pi.log_prob(abg)

            tau_pi = th.distributions.Categorical(logits=tau_logits)
            tau = tau_pi.sample()
            tau_log_prob = tau_pi.log_prob(tau)

            log_prob = abg_log_prob.sum(dim=1) + tau_log_prob

        abg = abg.cpu().numpy()
        tau = tau.cpu().numpy()
        value = value.cpu().numpy()
        log_prob = log_prob.cpu().numpy()
        act = self.make_action(abg, tau)
        return act, abg, tau, value, log_prob

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """根据观测值产生确定性动作，仅用于测试

        Args:
            obs (np.ndarray): 观测值

        Returns:
            np.ndarray: 确定性动作
        """
        obs = self.obs_to_tensor(obs)
        with th.no_grad():
            obs = obs.reshape(1, -1)
            abg, tau_logits, _ = self.ac_net(obs)
            tau = th.argmax(tau_logits, dim=1, keepdim=True)
        abg = abg.cpu().numpy()[0]
        tau = tau.cpu().numpy()[0][0]
        act = dict(abg=abg, tau=tau)
        return act

    def make_action(
        self, abgs: np.ndarray, taus: np.ndarray
    ) -> List[Dict[str, np.ndarray]]:
        acts = []
        for abg, tau in zip(abgs, taus):
            act = dict(abg=abg, tau=tau)
            acts.append(act)
        return acts

    def evaluate(
        self, obs: th.Tensor, abg: th.Tensor, tau: th.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        abg_logits, tau_logits, value = self.ac_net(obs)

        abg_std = self.abg_log_std.exp()
        abg_pi = th.distributions.Normal(abg_logits, abg_std)
        abg_log_prob = abg_pi.log_prob(abg)
        abg_entropy = abg_pi.entropy()

        tau_pi = th.distributions.Categorical(logits=tau_logits)
        tau = tau_pi.sample()
        tau_log_prob = tau_pi.log_prob(tau)
        tau_entropy = tau_pi.entropy()

        log_prob = abg_log_prob.sum(dim=1) + tau_log_prob
        entropy = abg_entropy.sum(dim=1) + tau_entropy

        return value, log_prob, entropy

    def obs_to_tensor(self, obs: np.ndarray) -> th.Tensor:
        return th.as_tensor(obs)

    def load(self, param_path):
        self.load_state_dict(th.load(param_path))


class ACNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        abg_dim: int,
        tau_dim: int,
    ):
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
        self.v_model = nn.Sequential(nn.Linear(64, 1))

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        feature = self.feature_model(obs)
        abg = self.abg_model(feature)
        tau = self.tau_model(feature)
        v = self.v_model(feature).reshape(-1)
        return abg, tau, v
