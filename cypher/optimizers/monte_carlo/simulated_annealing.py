# Author: Tanish Shinde
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from ..local import StochasticHillClimbingOptimizer


class SimulatedAnnealingOptimizer(StochasticHillClimbingOptimizer):
    def __init__(self, _main_args_, _opt_args_):
        super().__init__(_main_args_, _opt_args_)
        self.temp = 1

    # use _consider from StochasticHillClimbingOptimizer

    def _accept(self, _p_):
        return np.exp(-self._score_norm(_p_) / self.temp)

    def _iterate(self, i, _cand_, _p_):
        _cand_ = self._stochastic_hill_climb_iter(_cand_, _p_)

        self.temp = self.temp * self._opt_args_.annealing_rate

        return _cand_

    def _init_opt_positioner(self, _cand_):
        return super()._init_base_positioner(_cand_)
