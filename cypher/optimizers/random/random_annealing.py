# Author: Tanish Shinde
# Email: simon.blanke@yahoo.com
# License: MIT License


from ...base_optimizer import BaseOptimizer


class RandomAnnealingOptimizer(BaseOptimizer):
    def __init__(self, _main_args_, _opt_args_):
        super().__init__(_main_args_, _opt_args_)
        self.temp = 1

    def _iterate(self, i, _cand_, _p_):
        _p_.pos_new = _p_.move_climb(
            _cand_, _p_.pos_current, epsilon_mod=self.temp * self._opt_args_.epsilon_mod
        )
        _p_.score_new = _cand_.eval_pos(_p_.pos_new)

        if _p_.score_new > _cand_.score_best:
            _cand_, _p_ = self._update_pos(_cand_, _p_)

        self.temp = self.temp * self._opt_args_.annealing_rate

        return _cand_

    def _init_opt_positioner(self, _cand_):
        return super()._init_base_positioner(_cand_)
