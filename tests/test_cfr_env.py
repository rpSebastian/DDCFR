import numpy as np
import pytest

from ddcfr.cfr import DCFRSolver
from ddcfr.cfr.cfr_env import make_cfr_env, make_cfr_vec_env
from ddcfr.game.game_config import KuhnPoker
from ddcfr.utils.logger import Logger


def test_cfr_env():
    env = make_cfr_env("KuhnPoker")
    action = dict(abg=np.array([0.5, -0.5, 0.8], dtype=np.float64), tau=2)
    assert env.action_space.contains(action)
    s = env.reset()
    assert s[0] == 0
    for i in range(10):
        a = dict(abg=np.array([0.15, 0, 0.2], dtype=np.float64), tau=0)
        ns, r, d, i = env.step(a)
    assert d


def test_cfr_env_long():
    env = make_cfr_env(KuhnPoker())

    s = env.reset()
    while True:
        a = dict(abg=np.array([-0.4, 1, -0.2], dtype=np.float64), tau=0)
        ns, r, d, i = env.step(a)
        s = ns
        if d:
            break
    final_conv = i["conv"]
    expected_conv = 0.022778783925763657
    assert final_conv == pytest.approx(expected_conv, abs=1e-10)

    s = env.reset()
    a = dict(abg=np.array([-0.4, 1, -0.2], dtype=np.float64), tau=2)
    ns, r, d, i = env.step(a)
    a = dict(abg=np.array([-0.4, 1, -0.2], dtype=np.float64), tau=2)
    ns, r, d, i = env.step(a)
    expected_conv = 0.022778783925763657
    final_conv = i["conv"]
    assert final_conv == pytest.approx(expected_conv, abs=1e-10)


def test_cfr_env_space():
    env = make_cfr_env("KuhnPoker")

    assert env.observation_space.shape[0] == 2
    assert env.action_space["abg"].shape[0] == 3
    assert env.action_space["tau"].n == 5


@pytest.mark.skip(reason="take a long time, only for testing precesion once")
def test_cfr_precision():
    logger = Logger(writer_strings=[])
    game_config = KuhnPoker()
    env = make_cfr_env("KuhnPoker")
    s = env.reset()
    while True:
        a = dict(abg=np.array([0.15, 0, 0.2], dtype=np.float64), tau=0)
        ns, r, d, i = env.step(a)
        if d:
            break
    solver = DCFRSolver(game_config, logger)
    solver.learn(total_iterations=1000, eval_iterations_interval=100)
    conv = solver.conv_history[-1]
    print(conv, i)


def test_cfr_vec_env():
    env = make_cfr_vec_env([KuhnPoker(10), KuhnPoker(10)], n_envs=2)
    s = env.reset()
    assert len(s) == 2
    for i in range(10):
        a = [
            dict(abg=np.array([0.15, 0, 0.2], dtype=np.float64), tau=0),
            dict(abg=np.array([0.15, 0, 0.2], dtype=np.float64), tau=0),
        ]
        ns, r, d, i = env.step(a)
    assert d[0]
