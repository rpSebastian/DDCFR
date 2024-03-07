# Copyright (c) 2019 Eric Steinberger


import numpy as np

from PokerRL.cfr._CFRBase import CFRBase as _CFRBase


class DDCFR(_CFRBase):
    def __init__(
        self,
        model,
        total_iterations,
        name,
        chief_handle,
        game_cls,
        agent_bet_set,
        starting_stack_sizes=None,
        other_agent_bet_set=None,
        logger=None,
        game_name=None,
    ):
        super().__init__(
            name=name,
            chief_handle=chief_handle,
            game_cls=game_cls,
            starting_stack_sizes=starting_stack_sizes,
            agent_bet_set=agent_bet_set,
            other_agent_bet_set=other_agent_bet_set,
            algo_name="DuelingCFR",
        )
        self.model = model
        self.total_iterations = total_iterations
        self.tau_list = [1, 2, 5, 10, 20]
        self.alpha_range = [0, 5]
        self.beta_range = [-5, 0]
        self.gamma_range = [0, 5]
        self.logger = logger
        self.game_name = game_name
        self.reset()

    def iteration(self):
        if (self.tau == 0) and self.iters < self.total_iterations:
            state = self.get_state()
            action = self.model(state)
            self.alpha, self.beta, self.gamma, self.tau = self.unscale_action(action)
            self.logger.record(f"{self.game_name}/tau", self.tau)
        self.logger.record(f"{self.game_name}/alpha", self.alpha)
        self.logger.record(f"{self.game_name}/beta", self.beta)
        self.logger.record(f"{self.game_name}/gamma", self.gamma)
        super().iteration()
        self.tau -= 1
        self.iters += 1

    def reset(self):
        super().reset()
        self.iters = 0
        log_conv = np.log(self.expl / 1000) / np.log(10)
        self.start_log_conv = log_conv
        self.tau = 0

    def unscale_action(self, action):
        alpha, beta, gamma = action["abg"]
        tau = action["tau"]
        alpha = self.denormalize(alpha, *self.alpha_range)
        beta = self.denormalize(beta, *self.beta_range)
        gamma = self.denormalize(gamma, *self.gamma_range)
        tau = self.tau_list[tau]
        return alpha, beta, gamma, tau

    def denormalize(self, param, param_min, param_max):
        param_mid = (param_max + param_min) / 2
        param_half_len = (param_max - param_min) / 2
        param = param * param_half_len + param_mid
        return param

    def get_state(self):
        log_conv = np.log(self.expl / 1000) / np.log(10)
        iters = self._normalize_iters(self.iters)
        conv_frac = self.calc_conv_frac(log_conv)
        state = (iters, conv_frac)
        return np.array(state, dtype=np.float64)

    def calc_conv_frac(self, log_conv):
        start_log_conv = self.start_log_conv
        final_log_conv = -12
        conv_frac = (log_conv - final_log_conv) / (start_log_conv - final_log_conv)
        return conv_frac

    def _normalize_iters(self, iters):
        return iters / self.total_iterations

    def _regret_formula_after_first_it(self, ev_all_actions, strat_ev, last_regrets):
        imm_regrets = ev_all_actions - strat_ev
        regrets = last_regrets
        T = self._iter_counter + 1

        lt = (regrets <= 0).astype(np.float32)
        gt = 1 - lt
        k1 = (
            gt
            * regrets
            * np.power(T - 1, self.alpha)
            / (np.power(T - 1, self.alpha) + 1)
        )
        k2 = (
            lt * regrets * np.power(T - 1, self.beta) / (np.power(T - 1, self.beta) + 1)
        )
        regrets = k1 + k2 + imm_regrets

        return regrets

    def _regret_formula_first_it(self, ev_all_actions, strat_ev):
        return ev_all_actions - strat_ev

    def _compute_new_strategy(self, p_id):
        for t_idx in range(len(self._trees)):

            def _fill(_node):
                if _node.p_id_acting_next == p_id:
                    N = len(_node.children)
                    _capped_reg = np.maximum(_node.data["regret"], 0)
                    _reg_pos_sum = np.expand_dims(
                        np.sum(_capped_reg, axis=1), axis=1
                    ).repeat(N, axis=1)

                    with np.errstate(divide="ignore", invalid="ignore"):
                        _node.strategy = np.where(
                            _reg_pos_sum > 0.0,
                            _capped_reg / _reg_pos_sum,
                            np.full(
                                shape=(
                                    self._env_bldrs[t_idx].rules.RANGE_SIZE,
                                    N,
                                ),
                                fill_value=1.0 / N,
                                dtype=np.float32,
                            ),
                        )
                for c in _node.children:
                    _fill(c)

            _fill(self._trees[t_idx].root)

    def _add_strategy_to_average(self, p_id):
        def _fill(_node):
            if _node.p_id_acting_next == p_id:
                T = self._iter_counter + 1
                contrib = _node.strategy * np.expand_dims(
                    _node.reach_probs[p_id], axis=1
                )
                if self._iter_counter > 0:
                    _node.data["avg_strat_sum"] = (
                        _node.data["avg_strat_sum"] * np.power((T - 1) / T, self.gamma)
                        + contrib
                    )
                else:
                    _node.data["avg_strat_sum"] = contrib

                _s = np.expand_dims(np.sum(_node.data["avg_strat_sum"], axis=1), axis=1)

                with np.errstate(divide="ignore", invalid="ignore"):
                    _node.data["avg_strat"] = np.where(
                        _s == 0,
                        np.full(
                            shape=len(_node.allowed_actions),
                            fill_value=1.0 / len(_node.allowed_actions),
                        ),
                        _node.data["avg_strat_sum"] / _s,
                    )
                assert np.allclose(
                    np.sum(_node.data["avg_strat"], axis=1), 1, atol=0.0001
                )

            for c in _node.children:
                _fill(c)

        for t_idx in range(len(self._trees)):
            _fill(self._trees[t_idx].root)
