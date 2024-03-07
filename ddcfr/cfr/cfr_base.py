from itertools import count
from typing import List, Optional

import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability

from ddcfr.game.game_config import GameConfig
from ddcfr.utils.logger import Logger


class StateBase:
    def __init__(
        self,
        legal_actions: List[int],
    ):
        self.legal_actions = legal_actions
        self.num_actions = len(legal_actions)
        self._init_data()

    def _init_data(self):
        self.policy = {a: 1 / self.num_actions for a in self.legal_actions}
        self.regrets = {a: 0 for a in self.legal_actions}
        self.cum_policy = {a: 0 for a in self.legal_actions}

    def update_current_policy(self):
        regret_sum = 0
        for regret in self.regrets.values():
            regret_sum += max(0, regret)
        for a, regret in self.regrets.items():
            if regret_sum == 0:
                self.policy[a] = 1 / self.num_actions
            else:
                self.policy[a] = max(0, regret) / regret_sum

    def get_current_policy(self):
        policy = {}
        regret_sum = 0
        for regret in self.regrets.values():
            regret_sum += max(0, regret)
        for a, regret in self.regrets.items():
            if regret_sum == 0:
                policy[a] = 1 / self.num_actions
            else:
                policy[a] = max(0, regret) / regret_sum
        return policy

    def get_average_policy(self):
        cum_sum = sum(self.cum_policy.values())
        ave_policy = {}
        for a, cum in self.cum_policy.items():
            if cum_sum == 0:
                ave_policy[a] = 1 / self.num_actions
            else:
                ave_policy[a] = cum / cum_sum
        return ave_policy

    def init_uniform_policy(self):
        uniform_policy = {a: 1 / self.num_actions for a in self.legal_actions}
        return uniform_policy


class SolverBase:
    def __init__(
        self,
        game_config: GameConfig,
        logger: Logger,
    ):
        self.game_config = game_config
        self.logger = logger
        self.game = game_config.load_game()
        self.states = {}
        self.np = self.game.num_players()
        self.max_num_actions = self.game.num_distinct_actions()
        self.game_name = self.game_config.name
        self._init_states(self.game.new_initial_state())

        self.conv_history = []
        self.num_nodes_touched = 0
        self.num_iterations = 0

    def _init_states(self, h: pyspiel.State):
        if h.is_terminal():
            return
        if h.is_chance_node():
            for a in h.legal_actions():
                self._init_states(h.child(a))
            return
        self._lookup_state(h, h.current_player())
        for a in h.legal_actions():
            self._init_states(h.child(a))

    def _lookup_state(self, h: pyspiel.State, player: int):
        feature = h.information_state_string(player)
        feature = self._add_player_info_in_feature(feature, player)
        if self.states.get(feature) is None:
            self.states[feature] = self._init_state(h)
        return self.states[feature]

    def _add_player_info_in_feature(self, feature, player):
        feature = feature + "/" + str(player)
        return feature

    def _init_state(self, h):
        return StateBase(h.legal_actions())

    def learn(
        self,
        total_iterations: Optional[int] = None,
        total_nodes_touched: Optional[int] = None,
        eval_nodes_interval: Optional[int] = None,
        eval_iterations_interval: Optional[int] = None,
    ):
        if total_iterations:
            for self.num_iterations in range(0, total_iterations + 1):
                while (
                    self.num_nodes_touched
                    < self.game_config.num_nodes * 2 * self.num_iterations
                ):
                    self.iteration()
                self.after_iteration(
                    xlabel="iterations",
                    eval_nodes_interval=eval_nodes_interval,
                    eval_iterations_interval=eval_iterations_interval,
                )
        elif total_nodes_touched:
            for self.num_iterations in count(start=0, step=1):
                if self.num_iterations > 0:
                    self.iteration()
                self.after_iteration(
                    xlabel="nodes",
                    eval_nodes_interval=eval_nodes_interval,
                    eval_iterations_interval=eval_iterations_interval,
                )
                if self.num_nodes_touched >= total_nodes_touched:
                    break

    def iteration(self):
        return NotImplemented

    def after_iteration(
        self,
        xlabel: str,
        eval_nodes_interval: Optional[int] = None,
        eval_iterations_interval: Optional[int] = None,
    ):
        if self.num_iterations > 0:
            if (
                eval_iterations_interval
                and self.num_iterations % eval_iterations_interval != 0
            ):
                return
            if eval_nodes_interval and (
                self.num_nodes_touched - self.last_num_nodes_touched
                < eval_nodes_interval
            ):
                return
        self.last_num_nodes_touched = self.num_nodes_touched
        conv = self.calc_conv()
        self.conv_history.append(conv)
        self.logger.record(f"{self.game_name}/conv", conv)
        self.logger.record(f"{self.game_name}/iters", self.num_iterations)
        self.logger.record(f"{self.game_name}/nodes_touched", self.num_nodes_touched)
        if xlabel == "iterations":
            self.logger.dump(step=self.num_iterations)
        elif xlabel == "nodes":
            self.logger.dump(step=self.num_nodes_touched)
        else:
            raise ValueError(f"xlabel should be iterations or nodes, not {xlabel}")

    def calc_conv(self):
        conv = exploitability.exploitability(
            self.game,
            policy.tabular_policy_from_callable(self.game, self.average_policy()),
        )
        conv = max(conv, 1e-12)
        return conv

    def average_policy(self):
        def wrap(h):
            feature = h.information_state_string()
            player = h.current_player()
            feature = self._add_player_info_in_feature(feature, player)
            s = self.states[feature]
            return s.get_average_policy()

        return wrap
