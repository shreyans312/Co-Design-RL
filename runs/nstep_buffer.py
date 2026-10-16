"""N-step replay buffer for SB3 off-policy algorithms (single env).

Each stored transition is (s_t, a_t, sum_k gamma^k r_{t+k}, s_{t+n}, done). The matching TD target needs
gamma**n on the bootstrap term, so train the model with gamma=gamma**n while the buffer keeps the true gamma.
The last n-1 steps of an episode have no full n-step window and are not stored.
"""
from collections import deque

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


class NStepReplayBuffer(ReplayBuffer):
    def __init__(self, *args, n_step: int = 2, gamma: float = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_envs != 1:
            raise ValueError("NStepReplayBuffer supports a single environment")
        self.n_step = n_step
        self.nstep_gamma = gamma
        self._window = deque(maxlen=n_step)

    def add(self, obs, next_obs, action, reward, done, infos):
        self._window.append((obs, next_obs, action, np.asarray(reward, dtype=np.float32), done, infos))
        if len(self._window) == self.n_step:
            first, last = self._window[0], self._window[-1]
            n_step_return = sum(self.nstep_gamma ** k * w[3] for k, w in enumerate(self._window))
            super().add(first[0], last[1], first[2], n_step_return, last[4], last[5])
        if bool(np.any(done)):
            self._window.clear()
