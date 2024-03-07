from multiprocessing import Process
from pathlib import Path

from ddcfr.game.game_config import get_test_configs
from ddcfr.utils.logger import Logger
from ddcfr.utils.utils import load_module

game_configs = get_test_configs()
solver_names = ["CFR", "CFRPlus", "DCFR"]


def run(solver_name, game_config):
    solver_class = load_module(f"ddcfr.cfr:{solver_name}Solver")
    game_name = game_config.name
    folder = Path(__file__).absolute().parents[1] / "results" / solver_name / game_name
    folder.mkdir(exist_ok=True, parents=True)
    logger = Logger(
        writer_strings=["csv"],
        folder=folder,
    )
    solver = solver_class(game_config, logger)
    solver.learn(total_iterations=1000, eval_iterations_interval=1)
    logger.close()


process_list = []
for solver_name in solver_names:
    for game_config in game_configs:
        p = Process(target=run, args=(solver_name, game_config))
        p.start()
        process_list.append(p)

for p in process_list:
    p.join()
