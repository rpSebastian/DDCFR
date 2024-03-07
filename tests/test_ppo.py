import numpy as np
import torch as th

from ddcfr.cfr.cfr_env import make_cfr_vec_env
from ddcfr.game import KuhnPoker
from ddcfr.rl.ppo import A2CPolicy, ACNetwork, RolloutBuffer
from ddcfr.utils.utils import set_seed


def test_ac_network():
    set_seed(0)
    input_dim = 1
    abg_dim = 3
    tau_dim = 5
    ac_net = ACNetwork(input_dim, abg_dim, tau_dim)
    batch_size = 5
    obs = th.randn([batch_size, input_dim])
    abg_logits, tau_logits, value = ac_net(obs)
    assert abg_logits.shape == th.Size([batch_size, abg_dim])
    assert tau_logits.shape == th.Size([batch_size, tau_dim])
    assert value.shape == th.Size([batch_size])


def test_ac_policy():
    set_seed(0)
    env = make_cfr_vec_env([KuhnPoker(10)])
    policy = A2CPolicy(env.observation_space, env.action_space, th.device("cpu"), 1e-3)
    obs = env.reset()
    act, abg, tau, value, log_prob = policy.predict(obs)
    env.step(act)
    assert len(act) == 2
    assert act[0]["abg"].shape == th.Size([3])
    assert act[0]["tau"].shape == th.Size([])
    assert abg.shape == th.Size([2, 3])
    assert tau.shape == th.Size([2])
    assert log_prob.shape == th.Size([2])
    assert value.shape == th.Size([2])

    obs = obs_to_tensor(obs)
    abg = obs_to_tensor(abg)
    tau = obs_to_tensor(tau)
    value, log_prob, entropy = policy.evaluate(obs, abg, tau)
    assert value.size() == th.Size([2])
    assert log_prob.size() == th.Size([2])
    assert entropy.size() == th.Size([2])


def test_buffer():
    buffer_size = 10
    env = make_cfr_vec_env([KuhnPoker(10)])
    buffer = RolloutBuffer(
        buffer_size,
        env.observation_space,
        env.action_space,
        th.device("cpu"),
        gamma=0.99,
        gae_lambda=1.0,
    )
    policy = A2CPolicy(env.observation_space, env.action_space, th.device("cpu"), 1e-3)

    s = env.reset()
    act, abg, tau, value, log_prob = policy.predict(s)
    ns, r, d, i = env.step(act)
    for i in range(10):
        buffer.add(s, abg, tau, r, d, value, log_prob)
    buffer.compute_returns_and_advantage(0.5)

    samples = list(buffer.get())[0]
    assert samples.obs.size() == th.Size([20, 2])
    assert samples.acts_abg.size() == th.Size([20, 3])
    assert samples.acts_tau.size() == th.Size([20])
    assert samples.advs.size() == th.Size([20])
    assert samples.rets.size() == th.Size([20])
    assert samples.old_log_probs.size() == th.Size([20])
    assert samples.old_values.size() == th.Size([20])


def obs_to_tensor(obs: np.ndarray) -> th.Tensor:
    return th.as_tensor(obs)
