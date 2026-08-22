from copy import deepcopy

DEFAULT_CONFIG = {
    "simulation": {
        "timestep_hours": 1.0,
        "episode_hours": 168,
    },
    
    "design": {
        "pv_capacity_kwp": 6.0,
        "battery_capacity_kwh": 8.0,
    },
    
    "battery": {
        "efficiency": 0.90,
        "c_rate": 1.0,
        "minimum_soc_fraction": 0.0,
    },
    
    "ev": {
        "capacity_kwh": 80.0,
        "minimum_soc_fraction": 0.40,
        "maximum_power_kw": 5.0,
        "efficiency": 1.0,
        "minimum_duration_hours": 5,
        "maximum_duration_hours": 8,
        "weekday_visit_probability": 0.9,
        "weekend_visit_probability": 0.15,
        "arrival_hours": (7, 8, 9, 10, 11, 12, 13),
        "arrival_weights": (
            0.75, 0.90, 0.90, 0.75, 0.10, 0.10, 0.10
        ),
    },
    
    "economics": {
        "annual_discount_rate": 0.05,
        "pv_lifetime_years": 20,
        "pv_capex_fixed_chf": 100.0,
        "pv_capex_per_kwp_chf": 775.0,
        "pv_opex_per_kwp_year_chf": 100.0,
        "battery_lifetime_years": 10,
        "battery_capex_fixed_chf": 50.0,
        "battery_capex_per_kwh_chf": 300.0,
        "battery_opex_per_kwh_year_chf": 10.0,
        "ev_discharge_cost_per_kwh": 1.5,
        "ev_charge_revenue_per_kwh": 1.0,
        "include_ev_energy_cost": False,
    },
}

def get_default_config() -> dict: # Returns a deep copy of the default configuration dictionary
    return deepcopy(DEFAULT_CONFIG)

def _validate_probability(name: str, value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    
def _validate_efficiency(name: str, value: float) -> None:
    if not (0.0 < value <= 1.0):
        raise ValueError(f"{name} must be greater than 0 and less than or equal to 1, got {value}")

def validate_config(config: dict) -> None: # validate all model config values
    required_sections = {
        "simulation",
        "design",
        "battery",
        "ev",
        "economics",
    }
    
    missing_sections = required_sections - config.keys()
    
    if missing_sections:
        raise KeyError(f"Missing required sections in config: {missing_sections}")
    
    simulation = config["simulation"]
    design = config["design"]
    battery = config["battery"]
    ev = config["ev"]
    economics = config["economics"]
    
    if simulation["timestep_hours"] <= 0:
        raise ValueError("simulation.timestep_hours must be positive")
    
    if (not isinstance(simulation["episode_hours"], int) or isinstance(simulation["episode_hours"], bool)) or simulation["episode_hours"] <= 0:
        raise ValueError("simulation.episode_hours must be a positive number")
    
    if design["pv_capacity_kwp"] < 0:
        raise ValueError("design.pv_capacity_kwp must be non-negative")
    
    if design["battery_capacity_kwh"] < 0:
        raise ValueError("design.battery_capacity_kwh must be non-negative")
    
    _validate_efficiency("battery.efficiency", battery["efficiency"])
    
    if battery["c_rate"] <= 0:
        raise ValueError("battery.c_rate must be positive")
    
    _validate_probability("battery.minimum_soc_fraction", battery["minimum_soc_fraction"])
    
    if ev["capacity_kwh"] <= 0:
        raise ValueError("ev.capacity_kwh must be positive")
    
    if ev["maximum_power_kw"] <= 0:
        raise ValueError("ev.maximum_power_kw must be positive")
    
    _validate_efficiency("ev.efficiency", ev["efficiency"])
    
    _validate_probability("ev.minimum_soc_fraction", ev["minimum_soc_fraction"])
    
    if ev["minimum_duration_hours"] <= 0:
        raise ValueError("ev.minimum_duration_hours must be positive")
    
    if ev["maximum_duration_hours"] <= 0:
        raise ValueError("ev.maximum_duration_hours must be positive")
    
    if ev["minimum_duration_hours"] > ev["maximum_duration_hours"]:
        raise ValueError("ev.minimum_duration_hours cannot be greater than ev.maximum_duration_hours")
    
    _validate_probability("ev.weekday_visit_probability", ev["weekday_visit_probability"])
    _validate_probability("ev.weekend_visit_probability", ev["weekend_visit_probability"])
    
    arrival_hours = ev["arrival_hours"]
    arrival_weights = ev["arrival_weights"]
    
    if len(arrival_hours) != len(arrival_weights):
        raise ValueError("ev.arrival_hours and ev.arrival_weights must have the same length")
    
    if len(arrival_hours) == 0:
        raise ValueError("ev.arrival_hours and ev.arrival_weights cannot be empty")

    if any(hour < 0 or hour > 23 for hour in arrival_hours):
        raise ValueError("ev.arrival_hours must be between 0 and 23")
    
    if any(weight < 0 for weight in arrival_weights):
        raise ValueError("ev.arrival_weights must be non-negative")
    
    if sum(arrival_weights) <= 0:
        raise ValueError("ev.arrival_weights must sum to a positive value")
    
    if economics["annual_discount_rate"] < 0:
        raise ValueError("economics.annual_discount_rate must be non-negative")
    
    if economics["pv_lifetime_years"] <= 0:
        raise ValueError("economics.pv_lifetime_years must be positive")
    
    if economics["battery_lifetime_years"] <= 0:
        raise ValueError("economics.battery_lifetime_years must be positive")
    
    cost_parameters = [
        "pv_capex_fixed_chf",
        "pv_capex_per_kwp_chf",
        "pv_opex_per_kwp_year_chf",
        "battery_capex_fixed_chf",
        "battery_capex_per_kwh_chf",
        "battery_opex_per_kwh_year_chf",
        "ev_discharge_cost_per_kwh",
        "ev_charge_revenue_per_kwh",
    ]
    
    for parameter in cost_parameters:
        if economics[parameter] < 0:
            raise ValueError(f"economics.{parameter} must be non-negative")
        
    if not isinstance(economics["include_ev_energy_cost"], bool):
        raise ValueError("economics.include_ev_energy_cost must be a boolean")
    
def get_ev_minimum_soc_kwh(config: dict) -> float:
    ev = config["ev"]
    return ev["capacity_kwh"] * ev["minimum_soc_fraction"]

def get_battery_maximum_power_kw(config: dict) -> float:
    return config["design"]["battery_capacity_kwh"] * config["battery"]["c_rate"]

if __name__ == "__main__":
    config = get_default_config()
    validate_config(config)
    print("Configuration is valid.")
    print("EV minimum SOC (kWh):", get_ev_minimum_soc_kwh(config))
    print("Battery maximum power (kW):", get_battery_maximum_power_kw(config))