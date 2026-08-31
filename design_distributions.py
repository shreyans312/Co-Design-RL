import math

import torch
from torch import nn


class DesignMixture(nn.Module):
    def __init__(self, config: dict, seed: int) -> None:
        super().__init__()
        distribution = config["design_distribution"]
        self.name = distribution["name"]
        self.components = int(distribution["components"])
        self.dimensions = int(distribution["dimensions"])
        self.hard_assignment_log_prob = bool(distribution["hard_assignment_log_prob"])
        self.generator = torch.Generator().manual_seed(seed)

        low = torch.tensor(distribution["initial_low"], dtype=torch.float32)
        high = torch.tensor(distribution["initial_high"], dtype=torch.float32)
        minimum = torch.tensor(distribution["minimum"], dtype=torch.float32)
        maximum = torch.tensor(
            [math.inf if value is None else value for value in distribution["maximum"]],
            dtype=torch.float32,
        )
        self.register_buffer("minimum", minimum)
        self.register_buffer("maximum", maximum)
        self.register_buffer("active_mask", torch.ones(self.components, dtype=torch.bool))

        span = high - low
        mean_low = low + 0.1 * span
        mean_high = high - 0.1 * span
        random_means = torch.rand((self.components, self.dimensions), generator=self.generator)
        physical_means = mean_low + random_means * (mean_high - mean_low)

        if self.name == "gaussian":
            means = physical_means
            standard_deviations = span * distribution["initial_std_fraction"]
            standard_deviations = standard_deviations.expand_as(means)
        elif self.name == "lognormal":
            if "initial_parameter_low" in distribution:
                parameter_low = torch.tensor(distribution["initial_parameter_low"], dtype=torch.float32)
                parameter_high = torch.tensor(distribution["initial_parameter_high"], dtype=torch.float32)
                means = parameter_low + random_means * (parameter_high - parameter_low)
            else:
                means = torch.log(physical_means.clamp_min(1e-3))
            standard_deviations = torch.full_like(means, float(distribution["initial_std"]))
        else:
            raise ValueError(f"Unknown design distribution '{self.name}'")

        self.component_means = nn.Parameter(means)
        self.component_log_stds = nn.Parameter(torch.log(standard_deviations))
        self.mixture_logits = nn.Parameter(
            torch.zeros(self.components),
            requires_grad=bool(distribution["trainable_weights"]),
        )

    @property
    def active_components(self) -> int:
        return int(self.active_mask.sum().item())

    def component_parameters(self) -> dict:
        # Export mixture values for convergence plots
        with torch.no_grad():
            locations = self.component_means.detach().cpu().numpy().copy()
            scales = torch.exp(self.component_log_stds).detach().cpu().numpy().copy()
            weights = self._log_weights().exp().detach().cpu().numpy().copy()
            active = self.active_mask.detach().cpu().numpy().copy()
            return {
                "locations": locations,
                "scales": scales,
                "weights": weights,
                "active": active,
            }

    def _log_weights(self) -> torch.Tensor:
        masked_logits = self.mixture_logits.masked_fill(~self.active_mask, -torch.inf)
        return torch.log_softmax(masked_logits, dim=0)

    def _component_log_prob(self, latent: torch.Tensor) -> torch.Tensor:
        # Score samples against every mixture component
        log_stds = self.component_log_stds
        standard_deviations = torch.exp(log_stds)
        standardized = (latent[:, None, :] - self.component_means[None, :, :]) / standard_deviations[
            None, :, :
        ]
        log_prob = -0.5 * (
            standardized.square()
            + 2.0 * log_stds[None, :, :]
            + math.log(2.0 * math.pi)
        ).sum(dim=-1)
        if self.name == "lognormal":
            log_prob = log_prob - latent.sum(dim=-1, keepdim=True)
        return log_prob

    def _sample_components(self, components: torch.Tensor, generator: torch.Generator) -> dict:
        # Draw physical designs from chosen components
        batch_size = len(components)
        log_weights = self._log_weights()
        noise = torch.randn((batch_size, self.dimensions), generator=generator)

        with torch.no_grad():
            means = self.component_means[components]
            standard_deviations = torch.exp(self.component_log_stds[components])
            latent = means + standard_deviations * noise
            if self.name == "lognormal":
                designs = torch.exp(latent)
            else:
                designs = latent
            designs = torch.maximum(designs, self.minimum)
            designs = torch.minimum(designs, self.maximum)

        component_log_prob = self._component_log_prob(latent.detach())
        weighted_log_prob = component_log_prob + log_weights[None, :]
        if self.hard_assignment_log_prob:
            log_prob = weighted_log_prob.gather(1, components[:, None]).squeeze(1)
        else:
            log_prob = torch.logsumexp(weighted_log_prob, dim=1)

        return {
            "designs": designs,
            "log_prob": log_prob,
            "components": components,
        }

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> dict:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if generator is None:
            generator = self.generator

        components = torch.multinomial(
            self._log_weights().exp(),
            num_samples=batch_size,
            replacement=True,
            generator=generator,
        )
        return self._sample_components(components, generator)

    def sample_component(
        self,
        component: int,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> dict:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if component < 0 or component >= self.components:
            raise ValueError("component index is outside the mixture")
        if not self.active_mask[component]:
            raise ValueError("component is inactive")
        if generator is None:
            generator = self.generator

        components = torch.full((batch_size,), component, dtype=torch.long)
        return self._sample_components(components, generator)

    def mode(self) -> torch.Tensor:
        # Find the highest density design
        standard_deviations = torch.exp(self.component_log_stds)
        if self.name == "lognormal":
            candidates = torch.exp(self.component_means - standard_deviations.square())
            candidate_latent = torch.log(candidates)
            component_scores = self._component_log_prob(candidate_latent)
            density_scores = component_scores.diagonal()
        else:
            candidates = self.component_means
            density_scores = -(self.component_log_stds + 0.5 * math.log(2.0 * math.pi)).sum(dim=1)

        candidates = torch.maximum(candidates, self.minimum)
        candidates = torch.minimum(candidates, self.maximum)
        scores = density_scores + self._log_weights()
        return candidates[torch.argmax(scores)].detach().clone()

    def prune(self, component_scores: torch.Tensor) -> None:
        component_scores = torch.as_tensor(component_scores, dtype=torch.float32)
        if component_scores.shape != (self.components,):
            raise ValueError(f"component_scores must have shape ({self.components},)")
        if self.active_components <= 1:
            return

        # Keep the best half of active components
        active_indices = self.active_mask.nonzero(as_tuple=False).flatten()
        keep_count = max(1, len(active_indices) // 2)
        active_scores = component_scores[active_indices]
        keep_indices = active_indices[torch.topk(active_scores, keep_count).indices]
        next_mask = torch.zeros_like(self.active_mask)
        next_mask[keep_indices] = True
        self.active_mask.copy_(next_mask)


def create_design_distribution(config: dict, seed: int) -> DesignMixture:
    return DesignMixture(config=config, seed=seed)
