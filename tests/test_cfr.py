import pytest

from ddcfr.cfr import CFRSolver, DCFRSolver, DDCFRSolver
from ddcfr.game import KuhnPoker
from ddcfr.utils.logger import Logger


def test_solver():
    logger = Logger(writer_strings=[])
    game_config = KuhnPoker()
    solver = CFRSolver(game_config, logger)
    solver.learn(total_iterations=10)
    conv = solver.calc_conv()
    expected_conv = 0.0686987938171576
    assert conv == pytest.approx(expected_conv, abs=1e-10)

    game_config = KuhnPoker()
    solver = DCFRSolver(game_config, logger)
    solver.learn(total_iterations=10)
    conv = solver.calc_conv()
    expected_conv = 0.022778783925763657
    assert conv == pytest.approx(expected_conv, abs=1e-10)

    game_config = KuhnPoker()
    solver = DDCFRSolver(game_config, logger)
    solver.learn(total_iterations=10)
    conv = solver.calc_conv()
    expected_conv = 0.022778783925763657
    assert conv == pytest.approx(expected_conv, abs=1e-10)
