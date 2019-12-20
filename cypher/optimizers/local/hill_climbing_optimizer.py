# Author: Tanish Shinde
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from ...base_optimizer import BaseOptimizer


class HillClimbingOptimizer(BaseOptimizer):
    def __init__(self, _main_args_, _opt_args_):
        super().__init__(_main_args_, _opt_args_)

    def _hill_climb_iter(self, _cand_, _p_):
        score_new = -np.inf
        pos_new = None

        for _ in range(self._opt_args_.n_neighbours):
            _p_.pos_new = _p_.move_climb(_cand_, _p_.pos_current)
            _p_.score_new = _cand_.eval_pos(_p_.pos_new)

            if _p_.score_new > score_new:
                score_new = _p_.score_new
                pos_new = _p_.pos_new

        if score_new > _cand_.score_best:
            _p_.pos_new = pos_new
            _p_.score_new = score_new

            _cand_, _p_ = self._update_pos(_cand_, _p_)

        return _cand_, _p_

    def _iterate(self, i, _cand_, _p_):
        _cand_, _p_ = self._hill_climb_iter(_cand_, _p_)

        return _cand_

    def _init_opt_positioner(self, _cand_):
        return super()._init_base_positioner(_cand_)
