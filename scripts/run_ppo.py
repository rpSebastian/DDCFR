from pathlib import Path

from sacred.observers import FileStorageObserver

from ddcfr.game.game_config import get_train_configs
from ddcfr.rl.ppo import PPO
from ddcfr.utils.exp import ex
from ddcfr.utils.logger import Logger
from ddcfr.utils.utils import init_object, run_method


# flake8: noqa: C901
@ex.config
def config():
    train_game_configs = get_train_configs()
    learning_rate = 0.001
    n_steps = 256
    batch_size = 256
    n_epochs = 10
    n_envs = 20
    gamma = 0.99
    gae_lambda = 0.95
    ent_coef = 0
    vf_coef = 0.5
    max_grad_norm = 0.5
    clip_range = 0.2
    clip_range_cf = 0.0
    normalize_advantage = True
    seed = 0

    total_timesteps = 1e20
    log_interval = 5

    save_log = False
    if save_log:
        base_folder = Path(__file__).parents[1] / "logs" / "rl"
        ex.observers.append(FileStorageObserver(base_folder))


@ex.automain
def main(_config, _run):
    configs = dict(_config)

    if configs["save_log"]:
        writer_strings = ["stdout", "tensorboard", "sacred", "csv"]
        folder = configs["base_folder"] / str(_run._id)
        logger = Logger(
            writer_strings=writer_strings,
            folder=folder,
        )
    else:
        logger = Logger(
            writer_strings=["stdout"],
        )

    import numpy as np
    import torch

    np.set_printoptions(precision=10)
    torch.set_printoptions(precision=10)

    model = init_object(PPO, configs, logger=logger)
    run_method(model.learn, configs)
    logger.close()
