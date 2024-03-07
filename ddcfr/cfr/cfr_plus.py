from ddcfr.cfr.cfr import CFRSolver, CFRState


class CFRPlusState(CFRState):
    def cumulate_regret(self, T):
        for a in self.regrets.keys():
            self.regrets[a] = max(self.regrets[a] + self.imm_regrets[a], 0)

    def cumulate_policy(self, T):
        for a in self.regrets.keys():
            self.cum_policy[a] += T * self.policy[a] * self.reach


class CFRPlusSolver(CFRSolver):
    def _init_state(self, h):
        return CFRPlusState(h.legal_actions(), h.current_player())

    def update_state(self, s):
        s.cumulate_regret(self.num_iterations)

        s.cumulate_policy(self.num_iterations)

        s.update_current_policy()
