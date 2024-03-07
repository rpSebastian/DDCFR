import torch as th

from ddcfr.cfr.cfr_env import make_cfr_env
from ddcfr.es.cfr_model import CFRModel, CFRNetwork, CFRPolicy
from ddcfr.utils.utils import set_seed


def test_cfr_network():
    network = CFRNetwork(input_dim=1, abg_dim=3, tau_dim=5).to(th.double)

    x = th.tensor([-0.5], dtype=th.float64)
    abg, tau = network(x)
    assert abg.shape == th.Size([3])
    assert tau.shape == th.Size([5])

    x = th.tensor([[-0.5]], dtype=th.float64)
    abg, tau = network(x)
    assert abg.shape == th.Size([1, 3])
    assert tau.shape == th.Size([1, 5])


def test_cfr_policy():
    env = make_cfr_env("KuhnPoker")

    cfr_policy = CFRPolicy(env.observation_space, env.action_space, th.device("cpu"))
    obs = env.reset()
    action = cfr_policy.predict(obs)
    assert env.action_space.contains(action)


def test_cfr_model():
    set_seed(0)
    env = make_cfr_env("KuhnPoker")

    cfr_model = CFRModel(env.observation_space, env.action_space, th.device("cpu"))
    obs = env.reset()
    action = cfr_model(obs)
    assert env.action_space.contains(action)

    new_model = cfr_model.pertube(sigma=0.5)

    for [(k1, v1), (k2, v2)] in zip(cfr_model.params(), new_model.params()):
        dis = th.sum(th.square(v1 - v2)) / v1.reshape(-1).size()[0]
        assert k1 == k2
        assert dis < 0.5
