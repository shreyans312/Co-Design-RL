"""Callback that decays NormalActionNoise sigma exponentially: sigma_t = sigma0 * exp(-rate * t)."""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class NoiseDecay(BaseCallback):
    def __init__(self, sigma0: float, rate: float):
        super().__init__()
        self.sigma0 = sigma0
        self.rate = rate

    def _on_step(self) -> bool:
        sigma = self.sigma0 * float(np.exp(-self.rate * self.num_timesteps))
        self.model.action_noise._sigma[:] = sigma
        if self.n_calls % 500 == 0:
            self.logger.record("train/noise_std", sigma)
        return True
