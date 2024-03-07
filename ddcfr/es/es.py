import itertools
from typing import List, Optional

import numpy as np
import torch as th

from ddcfr.cfr.cfr_env import make_cfr_env
from ddcfr.es.cfr_eval import CFRModelVecEvaluator
from ddcfr.es.cfr_model import CFRModel
from ddcfr.es.optimizer import SGD, Adam
from ddcfr.game.game_config import GameConfig
from ddcfr.utils.logger import Logger
from ddcfr.utils.utils import set_seed


class EvolutionStategy:
    def __init__(
        self,
        train_game_configs: List[GameConfig],
        logger: Logger,
        save_freq: int = 5,
        num_epoches: int = 10000,
        sigma: float = 0.05,
        n: int = 100,
        num_train_evaluators: int = 200,
        lr: float = 0.01,
        use_mirror_sampling: bool = True,
        use_fitness_shaping: bool = True,
        use_adam: bool = True,
        seed: int = 0,
        device: str = "cpu",
        run_id: Optional[int] = None,
    ):
        self.set_seed(seed)
        self.num_epoches = num_epoches
        self.sigma = sigma
        self.n = n
        self.num_train_evaluators = num_train_evaluators
        self.lr = lr
        self.use_mirror_sampling = use_mirror_sampling
        self.use_fitness_shaping = use_fitness_shaping
        self.use_adam = use_adam
        self.train_game_configs = train_game_configs
        self.save_freq = save_freq
        self.logger = logger
        self.train_evaluator = CFRModelVecEvaluator(
            self.num_train_evaluators, verbose=True
        )
        self.run_id = run_id
        env = make_cfr_env("KuhnPoker")
        self.device = th.device(device)
        self.model = CFRModel(
            env.observation_space,
            env.action_space,
            self.device,
        )
        if self.use_adam:
            self.optim = Adam(self.model.params(), lr=self.lr)
        else:
            self.optim = SGD(self.model.params(), lr=self.lr)
        self.num_timesteps = 0
        self.train_game_steps = sum(
            game_config.iterations for game_config in self.train_game_configs
        )

    def train(self):
        for epoch in range(0, self.num_epoches + 1):
            if self.use_mirror_sampling:
                unevaluated_pertubed_models = list(
                    itertools.chain.from_iterable(
                        [self.model.mirror_pertube() for _ in range(self.n // 2)]
                    )
                ) + [self.model]
            else:
                unevaluated_pertubed_models = [
                    self.model.pertube() for _ in range(self.n)
                ] + [self.model]
            assert len(unevaluated_pertubed_models) == self.n + 1

            pertubed_models = self.train_evaluator.eval_models(
                unevaluated_pertubed_models, self.train_game_configs
            )
            pertubed_models.pop()
            self.num_timesteps += len(pertubed_models) * self.train_game_steps
            self.log_model(epoch)

            self.shape_model_score(pertubed_models)
            self.update(pertubed_models)

    def shape_model_score(self, pertubed_models):
        scores = [model.score for model in pertubed_models]
        if self.use_fitness_shaping:
            shaped_scores = self.fitness_shaping(scores)
        else:
            shaped_scores = scores
        for model, score in zip(pertubed_models, shaped_scores):
            model.score = score

    def fitness_shaping(self, scores):
        rank = np.argsort(np.argsort(-np.array(scores))) + 1
        lamda = len(scores)
        up = np.maximum(0, np.log(lamda / 2 + 1) - np.log(rank))
        down = np.sum(up)
        shaped_scores = up / down - 1 / lamda
        return shaped_scores

    def update(self, pertubed_models):
        grads = [[k, th.zeros_like(v)] for k, v, in self.model.params()]
        for pertubed_model in pertubed_models:
            eps_list = pertubed_model.eps_list
            score = pertubed_model.score
            for [(k1, eps), (k2, grad)] in zip(eps_list, grads):
                grad += 1 / (self.n * self.sigma) * (score * eps)
        self.optim.update(grads)

    def log_model(self, epoch):
        self.logger.record("epoch", epoch)
        self.logger.record("score", self.model.score)
        self.logger.record("num_timesteps", self.num_timesteps)
        for game_name, conv in self.model.convs.items():
            self.logger.record(game_name, conv)
        if epoch % self.save_freq == 0:
            self.logger.record("cfr_model", self.model)
        self.logger.dump(step=epoch)

    def set_seed(self, seed):
        set_seed(seed)
