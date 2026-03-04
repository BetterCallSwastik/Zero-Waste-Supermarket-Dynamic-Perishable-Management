"""
rl_env.py — Custom Gymnasium Environment for Perishable Pricing
===============================================================
Simulates a dynamic pricing game where an RL agent learns optimal
discount strategies to minimize food waste and maximize revenue.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


# Discount tiers mapped to action indices
DISCOUNT_MAP: Dict[int, float] = {
    0: 0.00,   # 0% off
    1: 0.10,   # 10% off
    2: 0.30,   # 30% off
    3: 0.50,   # 50% off
    4: 0.75,   # 75% off
}


class PerishablePricingEnv(gym.Env):
    """
    A Gymnasium environment that simulates perishable goods pricing.

    The agent decides what discount to apply each day. Higher discounts
    boost demand, selling more units. The goal is to clear stock before
    expiration while maximizing revenue.

    Observation: [normalized_days_until_expiry, normalized_stock, normalized_daily_demand]
    Action:      Discrete(5) — one of 5 discount tiers
    Reward:      Revenue from sales, with a heavy penalty for waste
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, render_mode: Optional[str] = None):
        super().__init__()

        # Validate required columns
        required_cols = ["product_name", "base_price", "days_until_expiry",
                         "initial_quantity", "daily_demand"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        self.df = df.copy()
        self.render_mode = render_mode

        # Normalization constants (used to scale observations to [0, 1])
        self.max_days = max(float(df["days_until_expiry"].max()), 1.0)
        self.max_stock = max(float(df["initial_quantity"].max()), 1.0)
        self.max_demand = max(float(df["daily_demand"].max()), 1.0)

        # --- Spaces ---
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(len(DISCOUNT_MAP))

        # Episode state (set in reset)
        self.current_product: Optional[pd.Series] = None
        self.days_left: int = 0
        self.stock: int = 0
        self.base_price: float = 0.0
        self.daily_demand: float = 0.0
        self.product_name: str = ""

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Sample a random product from the dataset
        idx = self.np_random.integers(0, len(self.df))
        row = self.df.iloc[idx]

        self.product_name = str(row["product_name"])
        self.base_price = float(row["base_price"])
        self.days_left = max(int(row["days_until_expiry"]), 1)
        self.stock = max(int(row["initial_quantity"]), 1)
        self.daily_demand = max(float(row["daily_demand"]), 1.0)

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        discount_pct = DISCOUNT_MAP[action]
        discounted_price = self.base_price * (1.0 - discount_pct)

        # --- Simulate demand boost from discount ---
        # Higher discounts multiplicatively increase the probability of sales
        demand_multiplier = 1.0 + discount_pct * 3.0  # e.g., 75% off → 3.25x demand
        effective_demand = int(self.daily_demand * demand_multiplier)

        # Add some stochasticity (±20%)
        noise = self.np_random.uniform(0.8, 1.2)
        quantity_sold = min(int(effective_demand * noise), self.stock)
        quantity_sold = max(quantity_sold, 0)

        # Update state
        self.stock -= quantity_sold
        self.days_left -= 1

        # --- Reward ---
        revenue = discounted_price * quantity_sold
        reward = revenue

        # Check termination
        terminated = False
        truncated = False

        if self.days_left <= 0 and self.stock > 0:
            # Waste penalty: unsold items at expiry
            waste_penalty = -100.0 * (self.stock / self.max_stock)
            reward += waste_penalty
            terminated = True
        elif self.stock <= 0:
            # All sold — bonus for clearing stock
            reward += 10.0
            terminated = True
        elif self.days_left <= 0:
            terminated = True

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        return np.array([
            self.days_left / self.max_days,
            self.stock / self.max_stock,
            self.daily_demand / self.max_demand,
        ], dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "product_name": self.product_name,
            "days_left": self.days_left,
            "stock": self.stock,
            "base_price": self.base_price,
        }
