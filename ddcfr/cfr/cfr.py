from typing import List

from ddcfr.cfr.cfr_base import SolverBase, StateBase
from ddcfr.game.game_config import GameConfig
from ddcfr.utils.logger import Logger


class CFRState(StateBase):
    def __init__(self, legal_actions: List[int], current_player: int):
        super().__init__(legal_actions)
        self.player = current_player

    def _init_data(self):
        super()._init_data()
        self.reach = 0
        self.imm_regrets = {a: 0 for a in self.legal_actions}

    def cumulate_regret(self):
        for a in self.regrets.keys():
            self.regrets[a] += self.imm_regrets[a]

    def cumulate_policy(self):
        for a, p in self.policy.items():
            self.cum_policy[a] += p * self.reach

    def clear_temp(self):
        for a in self.regrets.keys():
            self.imm_regrets[a] = 0
        self.reach = 0


class CFRSolver(SolverBase):
    def __init__(
        self,
        game_config: GameConfig,
        logger: Logger,
    ):
        super().__init__(game_config, logger)

    def _init_state(self, h):
        return CFRState(h.legal_actions(), h.current_player())

    def iteration(self):
        for i in range(self.np):
            h = self.game.new_initial_state()
            self.calc_regret(h, i, 1, 1)

            pending_states = [s for s in self.states.values() if s.player == i]

            for s in pending_states:
                self.update_state(s)
                s.clear_temp()

    def calc_regret(self, h, traveser, my_reach, opp_reach):
        self.num_nodes_touched += 1
        if h.is_terminal():
            return h.returns()[traveser]

        if h.is_chance_node():
            v = 0
            for a, p in h.chance_outcomes():
                v += p * self.calc_regret(h.child(a), traveser, my_reach, opp_reach * p)
            return v

        cur_player = h.current_player()
        s = self._lookup_state(h, cur_player)

        if cur_player != traveser:
            v = 0
            for a in h.legal_actions():
                p = s.policy[a]
                v += p * self.calc_regret(h.child(a), traveser, my_reach, opp_reach * p)
            return v

        child_v = {}
        v = 0
        for a in h.legal_actions():
            p = s.policy[a]
            child_v[a] = self.calc_regret(h.child(a), traveser, my_reach * p, opp_reach)
            v += p * child_v[a]

        for a in h.legal_actions():
            s.imm_regrets[a] += opp_reach * (child_v[a] - v)

        s.reach += my_reach
        return v

    def update_state(self, s):
        s.cumulate_regret()

        s.cumulate_policy()

        s.update_current_policy()
