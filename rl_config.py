from copy import deepcopy


COMMON_CONFIG = {
    "seed": 42,
    "observation": {
        "size": 11,
        "load_scale_kw": 10.0,
        "price_scale_chf_per_kwh": 1.0,
        "pv_design_scale_kwp": 12.0,
        "battery_design_scale_kwh": 20.0,
    },
    "design_distribution": {
        "dimensions": 2,
        "initial_low": (0.0, 0.0),
        "initial_high": (12.0, 20.0),
        "minimum": (0.0, 0.0),
        "maximum": (None, None),
        "learning_rate": 1e-3,
    },
    "training": {
        "iterations": 500,
        "episodes_per_iteration": 32,
        "validation_episodes": 32,
        "dataset_days": 365,
        "normalize_design_returns": True,
        "gradient_clip_norm": 0.5,
        "linear_learning_rate_decay": False,
    },
}


PPO_CONFIG = {
    "algorithm": {
        "name": "ppo",
        "hidden_sizes": (128, 128, 128),
        "activation": "tanh",
        "learning_rate": 1e-4,
        "discount_factor": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.0,
        "epochs": 4,
        "minibatch_size": 64,
    },
    "training": {
        "design_warmup_fraction": 0.10,
        "fixed_design_fraction": 0.10,
        "prune_interval_fraction": 0.10,
        "linear_learning_rate_decay": True,
    },
    "provenance": {
        "control_algorithm": "Schaff et al. paper and official nlimb code",
        "energy_environment": "local stochastic model",
    },
}


DDPG_CONFIG = {
    "algorithm": {
        "name": "ddpg",
        "hidden_sizes": (256, 256),
        "activation": "relu",
        "actor_learning_rate": 1e-4,
        "critic_learning_rate": 1e-3,
        "discount_factor": 0.99,
        "target_update_rate": 0.005,
        "replay_capacity": 100000,
        "minibatch_size": 256,
        "exploration_noise_std": 0.10,
        "learning_starts": 256,
    },
    "training": {
        "design_warmup_fraction": 0.0,
        "fixed_design_fraction": 0.0,
        "prune_interval_fraction": 0.0,
        "normalize_design_returns": False,
    },
    "provenance": {
        "control_algorithm": "PV co-optimisation paper",
        "energy_environment": "local stochastic model",
        "unspecified_ddpg_hyperparameters": "explicit implementation defaults",
    },
}


VARIANTS = {
    "ppo_gmm": {
        **PPO_CONFIG,
        "design_distribution": {
            "name": "gaussian",
            "components": 8,
            "initial_std_fraction": 0.2885,
            "maximum": (12.0, 20.0),
            "trainable_weights": False,
            "hard_assignment_log_prob": True,
            "prune_components": True,
            "prune_evaluations_per_component": 100,
            "entropy_weight": 0.0,
            "entropy_zero_fraction": 0.0,
        },
    },
    "ddpg_lognormal": {
        **DDPG_CONFIG,
        "design_distribution": {
            "name": "lognormal",
            "components": 3,
            "initial_parameter_low": (0.0, 0.0),
            "initial_parameter_high": (1.0, 1.0),
            "initial_std": 1.0,
            "trainable_weights": True,
            "hard_assignment_log_prob": False,
            "prune_components": False,
            "entropy_weight": 0.01,
            "entropy_zero_fraction": 0.50,
        },
    },
    "ppo_lognormal": {
        **PPO_CONFIG,
        "design_distribution": {
            "name": "lognormal",
            "components": 8,
            "initial_std": 0.75,
            "maximum": (12.0, 20.0),
            "trainable_weights": False,
            "hard_assignment_log_prob": True,
            "prune_components": True,
            "prune_evaluations_per_component": 100,
            "entropy_weight": 0.0,
            "entropy_zero_fraction": 0.0,
        },
    },
}


def _merge_nested(base: dict, update: dict) -> dict:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_nested(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def validate_rl_config(config: dict) -> None:
    algorithm = config["algorithm"]
    design = config["design_distribution"]
    training = config["training"]

    if algorithm["name"] not in {"ppo", "ddpg"}:
        raise ValueError("algorithm.name must be 'ppo' or 'ddpg'")
    if design["name"] not in {"gaussian", "lognormal"}:
        raise ValueError("design_distribution.name must be 'gaussian' or 'lognormal'")
    if design["components"] <= 0:
        raise ValueError("design_distribution.components must be positive")
    if training["iterations"] <= 0:
        raise ValueError("training.iterations must be positive")
    if training["episodes_per_iteration"] <= 0:
        raise ValueError("training.episodes_per_iteration must be positive")
    if training["validation_episodes"] <= 0:
        raise ValueError("training.validation_episodes must be positive")


def get_rl_config(model_name: str) -> dict:
    if model_name not in VARIANTS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {sorted(VARIANTS)}")

    config = _merge_nested(COMMON_CONFIG, VARIANTS[model_name])
    config["model_name"] = model_name
    validate_rl_config(config)
    return config
