import argparse
from multiprocessing import Process
from pathlib import Path

import torch as th

from ddcfr.cfr.cfr_env import make_cfr_env
from ddcfr.es.cfr_model import CFRModel
from ddcfr.game.game_config import get_test_configs
from ddcfr.rl.ppo import A2CPolicy
from ddcfr.utils.logger import Logger


def run(game_config, model, model_name):
    game_name = game_config.name
    folder = Path(__file__).absolute().parents[1] / "results" / model_name / game_name
    folder.mkdir(exist_ok=True, parents=True)
    logger = Logger(
        writer_strings=["csv"],
        folder=folder,
    )
    env = make_cfr_env(game_config, game_config.iterations, logger=logger)
    s = env.reset()
    while True:
        a = model(s)
        ns, r, d, i = env.step(a)
        s = ns
        if d:
            break
    logger.close()


def test(run_id, model_id, model_name, model_version):
    game_configs = get_test_configs()
    if model_name is None:
        run_path = (
            Path(__file__).absolute().parents[1]
            / "logs"
            / str(model_version)
            / str(run_id)
        )
        model_path = run_path / "model" / "cfr_model_{}.pkl".format(model_id)
        model_name = "DDCFR_{}_{}_{}".format(model_version, run_id, model_id)
    else:
        model_path = (
            Path(__file__).absolute().parents[1]
            / "models"
            / "{}.pkl".format(model_name)
        )
        model_name = "DDCFR_{}_{}".format(model_version, model_name)

    env = make_cfr_env("KuhnPoker")
    if model_version == "es":
        model = CFRModel(env.observation_space, env.action_space, th.device("cpu"))
    else:
        model = A2CPolicy(env.observation_space, env.action_space, th.device("cpu"))
    model.load(model_path)

    process_list = []
    for game_config in game_configs:
        p = Process(target=run, args=(game_config, model, model_name))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CFR in games")
    parser.add_argument("--run_id", type=int, help="run_id")
    parser.add_argument("--model_id", type=int, help="model_id")
    parser.add_argument("--model_name", type=str, help="model_name", default=None)
    parser.add_argument("--model_version", type=str, help="model_version", default="es")
    args = parser.parse_args()

    test(args.run_id, args.model_id, args.model_name, args.model_version)
