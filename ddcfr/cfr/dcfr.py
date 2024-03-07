from typing import List

import numpy as np

from ddcfr.cfr.cfr import CFRSolver, CFRState
from ddcfr.game.game_config import GameConfig
from ddcfr.utils.logger import Logger


class DCFRState(CFRState):
    def __init__(
        self,
        legal_actions: List[int],
        current_player: int,
        alpha: float = 1.5,
        beta: float = 0,
        gamma: float = 2,
    ):
        super().__init__(legal_actions, current_player)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def cumulate_regret(self, T, alpha, beta):
        T = float(T)
        for a in self.regrets.keys():
            if T == 1:
                self.regrets[a] = self.imm_regrets[a]
                continue
            if self.regrets[a] > 0:
                self.regrets[a] = (
                    self.regrets[a]
                    * (np.power(T - 1, alpha) / (np.power(T - 1, alpha) + 1))
                    + self.imm_regrets[a]
                )
            else:
                self.regrets[a] = (
                    self.regrets[a]
                    * (np.power(T - 1, beta) / (np.power(T - 1, beta) + 1))
                    + self.imm_regrets[a]
                )

    def cumulate_policy(self, T, gamma):
        T = float(T)
        for a in self.regrets.keys():
            if T == 1:
                self.cum_policy[a] = self.reach * self.policy[a]
                continue
            self.cum_policy[a] = (
                self.cum_policy[a] * np.power((T - 1) / T, gamma)
                + self.reach * self.policy[a]
            )


class DCFRSolver(CFRSolver):
    def __init__(
        self,
        game_config: GameConfig,
        logger: Logger,
        alpha: float = 1.5,
        beta: float = 0,
        gamma: float = 2,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        super().__init__(game_config, logger)

    def _init_state(self, h):
        return DCFRState(
            h.legal_actions(),
            h.current_player(),
            self.alpha,
            self.beta,
            self.gamma,
        )

    def update_state(self, s):
        s.cumulate_regret(self.num_iterations, s.alpha, s.beta)

        s.cumulate_policy(self.num_iterations, s.gamma)

        s.update_current_policy()
