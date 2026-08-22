from copy import deepcopy
from typing import Callable

import numpy as np
import pandas as pd

from config import get_default_config, validate_config
from data import DEFAULT_YEAR, generate_dataset
from stochastic import generate_episode_scenario, validate_episode_scenario


def capital_recovery_factor(discount_rate: float, lifetime_years: int) -> float:
    if lifetime_years <= 0:
        raise ValueError("lifetime_years must be positive")
    
    if discount_rate < 0:
        raise ValueError("discount_rate cannot be negative")
    
    if discount_rate == 0:
        return 1.0 / lifetime_years
    
    growth = (1.0 + discount_rate) ** lifetime_years
    
    return discount_rate * growth / (growth - 1.0)


def calculate_hourly_design_cost(config: dict) -> float:
    validate_config(config)
    
    design = config["design"]
    economics = config["economics"]
    
    pv_capacity = design["pv_capacity_kwp"]
    battery_capacity = design["battery_capacity_kwh"]
    
    pv_installed = 1.0 if pv_capacity > 0 else 0.0
    battery_installed = 1.0 if battery_capacity > 0 else 0.0
    
    pv_capex = (
        economics["pv_capex_fixed_chf"] * pv_installed
        + economics["pv_capex_per_kwp_chf"] * pv_capacity
    )
    
    battery_capex = (
        economics["battery_capex_fixed_chf"] * battery_installed
        + economics["battery_capex_per_kwh_chf"] * battery_capacity
    )
    
    pv_recovery_factor = capital_recovery_factor(
        discount_rate=economics["annual_discount_rate"],
        lifetime_years=economics["pv_lifetime_years"],
    )
    
    battery_recovery_factor = capital_recovery_factor(
        discount_rate=economics["annual_discount_rate"],
        lifetime_years=economics["battery_lifetime_years"],
    )
    
    annualized_pv_capex = pv_capex * pv_recovery_factor
    annualized_battery_capex = battery_capex * battery_recovery_factor
    
    annual_pv_opex = economics["pv_opex_per_kwp_year_chf"] * pv_capacity
    annual_battery_opex = economics["battery_opex_per_kwh_year_chf"] * battery_capacity
    
    total_annual_cost = (annualized_pv_capex + annualized_battery_capex 
                         + annual_pv_opex + annual_battery_opex
    )
    
    return total_annual_cost / 8760.0


def apply_storage_power(requested_power_kw: float, soc_kwh: float, capacity_kwh: float,
                        minimum_soc_kwh: float, maximum_power_kw: float, efficiency: float,
                        timestep_hours: float,) -> tuple[float, float]:
    if capacity_kwh <= 0:
        return 0.0, 0.0
    
    requested_power_kw = float(
        np.clip(requested_power_kw, -maximum_power_kw, maximum_power_kw)
    )
    
    if requested_power_kw >= 0:
        available_discharge_kw = (
            max(0.0, soc_kwh - minimum_soc_kwh) * efficiency / timestep_hours
        )
        
        actual_power_kw = min(
            requested_power_kw,
            maximum_power_kw,
            available_discharge_kw,
        )
        
        next_soc_kwh = (
            soc_kwh - actual_power_kw * timestep_hours / efficiency
        )
    else:
        requested_charge_kw = -requested_power_kw
        available_charge_kw = (
            max(0.0, capacity_kwh - soc_kwh) / (efficiency * timestep_hours)
        )
        
        actual_charge_kw = min(
            requested_charge_kw,
            maximum_power_kw,
            available_charge_kw,
        )
        
        actual_power_kw = -actual_charge_kw
        next_soc_kwh = (
            soc_kwh + actual_charge_kw * timestep_hours * efficiency
        )
    
    next_soc_kwh = float(
        np.clip(next_soc_kwh, minimum_soc_kwh, capacity_kwh)
    )
    
    return float(actual_power_kw), next_soc_kwh


class EnergySystemSimulator:
    def __init__(self, config: dict) -> None:
        validate_config(config)
        self.config = deepcopy(config)
        self.scenario = None
        self.episode_data = None
        self.ev_schedule = None
        
        self.current_step = 0
        self.battery_soc_kwh = 0.0
        self.ev_soc_kwh = 0.0
        self.current_ev_session_id = -1
        
        self.cumulative_cost_chf = 0.0
        self.cumulative_reward = 0.0
        
        self.done = True
        self.history = []
    
    def reset(self, scenario: dict) -> dict:
        validate_episode_scenario(scenario=scenario, config=self.config)
        
        self.scenario = scenario
        self.episode_data = (
            scenario["episode_data"].copy().reset_index(drop=True)
        )
        self.ev_schedule = (
            scenario["ev_schedule"].copy().reset_index(drop=True)
        )
        
        self.current_step = 0
        self.battery_soc_kwh = float(
            scenario["initial_battery_soc_kwh"]
        )
        self.ev_soc_kwh = 0.0
        self.current_ev_session_id = -1
        
        self.cumulative_cost_chf = 0.0
        self.cumulative_reward = 0.0
        
        self.done = False
        self.history = []
        
        self._synchronize_ev_state()
        
        return self.get_state()
    
    def _synchronize_ev_state(self) -> None:
        if self.done:
            self.ev_soc_kwh = 0.0
            self.current_ev_session_id = -1
            return
        
        ev_row = self.ev_schedule.iloc[self.current_step]
        ev_present = bool(ev_row["ev_present"])
        ev_arrival = bool(ev_row["ev_arrival"])
        session_id = int(ev_row["ev_session_id"])
        
        if ev_arrival:
            self.ev_soc_kwh = float(ev_row["ev_arrival_soc_kwh"])
            self.current_ev_session_id = session_id
        elif not ev_present:
            self.ev_soc_kwh = 0.0
            self.current_ev_session_id = -1
        elif self.current_ev_session_id != session_id:
            raise RuntimeError("EV session continued without an arrival state")
    
    def get_state(self) -> dict:
        if self.done:
            return {
                "step": self.current_step,
                "done": True,
                "battery_soc_kwh": self.battery_soc_kwh,
                "ev_present": False,
                "ev_soc_kwh": 0.0,
                "cumulative_cost_chf": self.cumulative_cost_chf,
                "cumulative_reward": self.cumulative_reward,
            }
        
        data_row = self.episode_data.iloc[self.current_step]
        ev_row = self.ev_schedule.iloc[self.current_step]
        
        pv_power_kw = (
            float(data_row["pv_per_kwp"]) * self.config["design"]["pv_capacity_kwp"]
        )
        
        return {
            "step": self.current_step,
            "done": False,
            "timestamp": data_row["timestamp"],
            "hour": int(data_row["hour"]),
            "weekday": int(data_row["weekday"]),
            "pv_power_kw": pv_power_kw,
            "load_kw": float(data_row["load_kw"]),
            "import_price": float(data_row["import_price"]),
            "export_price": float(data_row["export_price"]),
            "battery_soc_kwh": self.battery_soc_kwh,
            "battery_capacity_kwh": self.config["design"]["battery_capacity_kwh"],
            "ev_present": bool(ev_row["ev_present"]),
            "ev_session_id": int(ev_row["ev_session_id"]),
            "ev_soc_kwh": self.ev_soc_kwh,
            "ev_capacity_kwh": self.config["ev"]["capacity_kwh"],
            "cumulative_cost_chf": self.cumulative_cost_chf,
            "cumulative_reward": self.cumulative_reward,
        }
    
    def _calculate_costs(self, grid_import_kw: float, grid_export_kw: float,
                        ev_power_kw: float, import_price: float,
                        export_price: float,) -> dict:
        timestep_hours = self.config["simulation"]["timestep_hours"]
        economics = self.config["economics"]
        
        grid_energy_cost_chf = timestep_hours * (
            grid_import_kw * import_price - grid_export_kw * export_price
        )
        
        design_cost_chf = (
            calculate_hourly_design_cost(self.config) * timestep_hours
        )
        
        ev_energy_cost_chf = 0.0
        if economics["include_ev_energy_cost"]:
            ev_discharge_kwh = max(0.0, ev_power_kw) * timestep_hours
            ev_charge_kwh = max(0.0, -ev_power_kw) * timestep_hours
            
            ev_energy_cost_chf = (
                ev_discharge_kwh
                * economics["ev_discharge_cost_per_kwh"]
                - ev_charge_kwh
                * economics["ev_charge_revenue_per_kwh"]
            )
        
        total_cost_chf = (
            grid_energy_cost_chf
            + design_cost_chf
            + ev_energy_cost_chf
        )
        
        return {
            "grid_energy_cost_chf": grid_energy_cost_chf,
            "design_cost_chf": design_cost_chf,
            "ev_energy_cost_chf": ev_energy_cost_chf,
            "total_cost_chf": total_cost_chf,
        }
    
    def step(
        self,
        action: list[float] | tuple[float, float] | np.ndarray,
    ) -> tuple[dict | None, float, bool, dict]:
        if self.done:
            raise RuntimeError("Cannot call step after the episode has ended")
        
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (2,):
            raise ValueError("action must contain two values")
        
        if not np.isfinite(action_array).all():
            raise ValueError("action values must be finite")
        
        action_array = np.clip(action_array, -1.0, 1.0)
        battery_action = float(action_array[0])
        ev_action = float(action_array[1])
        
        data_row = self.episode_data.iloc[self.current_step]
        ev_row = self.ev_schedule.iloc[self.current_step]
        timestep_hours = self.config["simulation"]["timestep_hours"]
        
        battery_capacity_kwh = self.config["design"]["battery_capacity_kwh"]
        battery_minimum_soc_kwh = (
            battery_capacity_kwh
            * self.config["battery"]["minimum_soc_fraction"]
        )
        battery_maximum_power_kw = (
            battery_capacity_kwh
            * self.config["battery"]["c_rate"]
        )
        
        requested_battery_power_kw = (
            battery_action * battery_maximum_power_kw
        )
        battery_power_kw, next_battery_soc_kwh = apply_storage_power(
            requested_power_kw=requested_battery_power_kw,
            soc_kwh=self.battery_soc_kwh,
            capacity_kwh=battery_capacity_kwh,
            minimum_soc_kwh=battery_minimum_soc_kwh,
            maximum_power_kw=battery_maximum_power_kw,
            efficiency=self.config["battery"]["efficiency"],
            timestep_hours=timestep_hours,
        )
        
        ev_present = bool(ev_row["ev_present"])
        requested_ev_power_kw = 0.0
        ev_power_kw = 0.0
        next_ev_soc_kwh = self.ev_soc_kwh
        
        if ev_present:
            ev_capacity_kwh = self.config["ev"]["capacity_kwh"]
            ev_minimum_soc_kwh = (
                ev_capacity_kwh
                * self.config["ev"]["minimum_soc_fraction"]
            )
            ev_maximum_power_kw = self.config["ev"]["maximum_power_kw"]
            requested_ev_power_kw = ev_action * ev_maximum_power_kw
            
            ev_power_kw, next_ev_soc_kwh = apply_storage_power(
                requested_power_kw=requested_ev_power_kw,
                soc_kwh=self.ev_soc_kwh,
                capacity_kwh=ev_capacity_kwh,
                minimum_soc_kwh=ev_minimum_soc_kwh,
                maximum_power_kw=ev_maximum_power_kw,
                efficiency=self.config["ev"]["efficiency"],
                timestep_hours=timestep_hours,
            )
        
        pv_power_kw = (
            float(data_row["pv_per_kwp"])
            * self.config["design"]["pv_capacity_kwp"]
        )
        load_kw = float(data_row["load_kw"])
        
        net_grid_power_kw = (
            load_kw
            - pv_power_kw
            - battery_power_kw
            - ev_power_kw
        )
        grid_import_kw = max(0.0, net_grid_power_kw)
        grid_export_kw = max(0.0, -net_grid_power_kw)
        
        costs = self._calculate_costs(
            grid_import_kw=grid_import_kw,
            grid_export_kw=grid_export_kw,
            ev_power_kw=ev_power_kw,
            import_price=float(data_row["import_price"]),
            export_price=float(data_row["export_price"]),
        )
        
        reward = -costs["total_cost_chf"]
        self.battery_soc_kwh = next_battery_soc_kwh
        self.ev_soc_kwh = next_ev_soc_kwh
        self.cumulative_cost_chf += costs["total_cost_chf"]
        self.cumulative_reward += reward
        
        power_balance_error_kw = (
            grid_import_kw
            - grid_export_kw
            + pv_power_kw
            + battery_power_kw
            + ev_power_kw
            - load_kw
        )
        
        transition = {
            "step": self.current_step,
            "timestamp": data_row["timestamp"],
            "battery_action": battery_action,
            "ev_action": ev_action,
            "requested_battery_power_kw": requested_battery_power_kw,
            "battery_power_kw": battery_power_kw,
            "battery_soc_kwh": self.battery_soc_kwh,
            "ev_present": ev_present,
            "ev_session_id": int(ev_row["ev_session_id"]),
            "requested_ev_power_kw": requested_ev_power_kw,
            "ev_power_kw": ev_power_kw,
            "ev_soc_kwh": self.ev_soc_kwh,
            "pv_power_kw": pv_power_kw,
            "load_kw": load_kw,
            "grid_import_kw": grid_import_kw,
            "grid_export_kw": grid_export_kw,
            "power_balance_error_kw": power_balance_error_kw,
            **costs,
            "reward": reward,
            "cumulative_cost_chf": self.cumulative_cost_chf,
            "cumulative_reward": self.cumulative_reward,
        }
        
        self.history.append(transition)
        self.current_step += 1
        self.done = self.current_step >= len(self.episode_data)
        
        if self.done:
            next_state = None
        else:
            self._synchronize_ev_state()
            next_state = self.get_state()
        
        return next_state, reward, self.done, transition
    
    def run_episode(
        self,
        policy: Callable[[dict], list[float] | tuple[float, float] | np.ndarray],
    ) -> pd.DataFrame:
        if self.done:
            raise RuntimeError("Reset the simulator before running an episode")
        
        while not self.done:
            state = self.get_state()
            action = policy(state)
            self.step(action)
        
        return pd.DataFrame(self.history)


if __name__ == "__main__":
    config = get_default_config()
    dataset = generate_dataset(
        year=DEFAULT_YEAR,
        days=365,
        seed=42,
    )
    
    scenario = generate_episode_scenario(
        data=dataset,
        config=config,
        split="train",
        seed=42,
    )
    
    simulator = EnergySystemSimulator(config=config)
    simulator.reset(scenario=scenario)
    
    def idle_policy(state: dict) -> np.ndarray:
        return np.array([0.0, 0.0], dtype=float)
    
    results = simulator.run_episode(policy=idle_policy)
    
    print("Episode steps:", len(results))
    print("Total episode cost:", round(results["total_cost_chf"].sum(), 2), "CHF")
    print(
        "Maximum power balance error:",
        results["power_balance_error_kw"].abs().max(),
        "kW",
    )
