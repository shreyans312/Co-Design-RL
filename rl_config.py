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
        "scenarios_per_design": 1,
        "validation_episodes": 32,
        "dataset_days": 365,
        "normalize_design_returns": True,
        "design_return_baseline": "none",
        "design_scoring": "training",
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
        "exploration_noise_std": 0.20,
        "learning_starts": 512,
    },
    "training": {
        "design_warmup_fraction": 0.0,
        "fixed_design_fraction": 0.0,
        "prune_interval_fraction": 0.0,
        "normalize_design_returns": False,
        "design_return_baseline": "leave_one_out",
    },
    "provenance": {
        "control_algorithm": "PV co-optimisation paper",
        "energy_environment": "local stochastic model",
        "unspecified_ddpg_hyperparameters": "explicit implementation defaults",
        "design_initial_std": "1.5 implementation choice; paper specifies high variance without a value",
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
            "initial_parameter_low": (0.1823, 0.6931),
            "initial_parameter_high": (2.3795, 2.8904),
            "initial_std": 3.0,
            "learning_rate": 5e-3,
            "trainable_weights": True,
            "hard_assignment_log_prob": False,
            "prune_components": False,
            "entropy_weight": 5,
            "entropy_zero_fraction": 0.50,
        },
    },
    "ddpg_lognormal_improved": {
        **DDPG_CONFIG,
        "algorithm": {
            **DDPG_CONFIG["algorithm"],
            "exploration_noise_std": 0.20,
            "final_exploration_noise_std": 0.05,
        },
        "training": {
            **DDPG_CONFIG["training"],
            "scenarios_per_design": 4,
            "normalize_design_returns": True,
        },
        "design_distribution": {
            "name": "lognormal",
            "components": 4,
            "initial_std": 0.55,
            "minimum_std": 0.15,
            "maximum_std": 0.90,
            "minimum": (0.0, 0.0),
            "maximum": (12.0, 20.0),
            "stratified_initialization": True,
            "stratified_sampling": True,
            "minimum_component_weight": 0.05,
            "trainable_weights": True,
            "hard_assignment_log_prob": False,
            "prune_components": False,
            "entropy_weight": 0.02,
            "entropy_zero_fraction": 0.75,
        },
        "provenance": {
            "control_algorithm": "DDPG",
            "energy_environment": "local stochastic model",
            "design_method": "bounded stratified log-normal mixture with common random scenarios",
            "paper_faithful": False,
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
    if training["scenarios_per_design"] <= 0:
        raise ValueError("training.scenarios_per_design must be positive")
    if training["episodes_per_iteration"] % training["scenarios_per_design"]:
        raise ValueError("episodes_per_iteration must be divisible by scenarios_per_design")
    if training["validation_episodes"] <= 0:
        raise ValueError("training.validation_episodes must be positive")
    if training["design_return_baseline"] not in {"none", "leave_one_out"}:
        raise ValueError("training.design_return_baseline must be 'none' or 'leave_one_out'")
    if training.get("design_scoring", "training") not in {"training", "frozen_actor"}:
        raise ValueError("training.design_scoring must be 'training' or 'frozen_actor'")
    if training.get("design_scoring", "training") == "frozen_actor" and algorithm["name"] != "ddpg":
        raise ValueError("Frozen-actor design scoring is a DDPG experiment")
    minimum_weight = float(design.get("minimum_component_weight", 0.0))
    if minimum_weight < 0.0 or minimum_weight * design["components"] >= 1.0:
        raise ValueError("minimum_component_weight must be non-negative and leave trainable mass")
    minimum_std = float(design.get("minimum_std", 0.0))
    maximum_std = float(design.get("maximum_std", float("inf")))
    if minimum_std < 0.0 or maximum_std <= 0.0 or minimum_std > maximum_std:
        raise ValueError("design standard-deviation bounds are invalid")


def get_rl_config(model_name: str) -> dict:
    if model_name not in VARIANTS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {sorted(VARIANTS)}")

    config = _merge_nested(COMMON_CONFIG, VARIANTS[model_name])
    config["model_name"] = model_name
    validate_rl_config(config)
    return config
