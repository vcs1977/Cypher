# Author: Tanish Shinde
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np

from ...base_optimizer import BaseOptimizer
from ...base_positioner import BasePositioner


class ParticleSwarmOptimizer(BaseOptimizer):
    def __init__(self, _main_args_, _opt_args_):
        super().__init__(_main_args_, _opt_args_)

    def _init_particles(self, _cand_):
        _p_list_ = [Particle() for _ in range(self._opt_args_.n_particles)]
        for i, _p_ in enumerate(_p_list_):
            _p_.nr = i
            _p_.pos_current = _cand_._space_.get_random_pos()
            _p_.pos_best = _p_.pos_current
            _p_.velo = np.zeros(len(_cand_._space_.para_space))

        return _p_list_

    def _move_positioners(self, _cand_, _p_list_):
        for _p_ in _p_list_:
            r1, r2 = random.random(), random.random()

            A = self._opt_args_.inertia * _p_.velo
            B = (
                self._opt_args_.cognitive_weight
                * r1
                * np.subtract(_p_.pos_best, _p_.pos_current)
            )
            C = (
                self._opt_args_.social_weight
                * r2
                * np.subtract(_cand_.pos_best, _p_.pos_current)
            )

            new_velocity = A + B + C

            _p_.velo = new_velocity
            _p_.pos_new = _p_.move_part(_cand_, _p_.pos_current)

    def _eval_particles(self, _cand_, _p_list_):
        for _p_ in _p_list_:
            _p_.score_new = _cand_.eval_pos(_p_.pos_new)

            if _p_.score_new > _cand_.score_best:
                _cand_, _p_ = self._update_pos(_cand_, _p_)

    def _iterate(self, i, _cand_, _p_list_):
        self._move_positioners(_cand_, _p_list_)
        self._eval_particles(_cand_, _p_list_)

        return _cand_

    def _init_opt_positioner(self, _cand_):
        _p_list_ = self._init_particles(_cand_)

        return _p_list_


class Particle(BasePositioner):
    def __init__(self):
        self.nr = None
        self.velo = None

    def move_part(self, _cand_, pos):
        pos_new = (pos + self.velo).astype(int)
        # limit movement
        n_zeros = [0] * len(_cand_._space_.dim)
        return np.clip(pos_new, n_zeros, _cand_._space_.dim)
