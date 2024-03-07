from pathlib import Path

from sacred.observers import FileStorageObserver

from ddcfr.es.es import EvolutionStategy
from ddcfr.game.game_config import get_train_configs
from ddcfr.utils.exp import ex
from ddcfr.utils.logger import Logger
from ddcfr.utils.utils import init_object, run_method


@ex.config
def config():
    num_train_games = 4
    train_game_configs = get_train_configs(num_train_games)
    save_freq = 5
    num_epoches = 1000
    sigma = 0.05
    n = 100
    num_train_evaluators = 200
    num_test_evaluators = 50
    lr = 0.01
    use_mirror_sampling = True
    use_fitness_shaping = True
    use_adam = True
    seed = 0

    save_log = False
    if save_log:
        base_folder = Path(__file__).parents[1] / "logs" / "es"
        ex.observers.append(FileStorageObserver(base_folder))


@ex.automain
def main(_config, _run):
    configs = dict(_config)
    print(configs["train_game_configs"])

    if configs["save_log"]:
        writer_strings = ["stdout", "tensorboard", "sacred", "csv"]
        folder = configs["base_folder"] / str(_run._id)
        logger = Logger(
            writer_strings=writer_strings,
            folder=folder,
        )
        run_id = _run._id
    else:
        logger = Logger(
            writer_strings=["stdout"],
        )
        run_id = None
    es = init_object(EvolutionStategy, configs, logger=logger, run_id=run_id)
    run_method(es.train, configs)
    logger.close()
