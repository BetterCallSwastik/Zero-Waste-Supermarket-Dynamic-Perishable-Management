"""
train_agent.py — DQN Training & Inference
==========================================
Loads the perishable goods dataset, trains a DQN agent using
stable-baselines3, saves the model, and exposes a prediction function.
"""

import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from rl_env import PerishablePricingEnv, DISCOUNT_MAP

# ---------------------
# Config
# ---------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "perishable_goods_management.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "dqn_pricing_agent")
TIMESTEPS = 10_000


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load and validate the perishable goods dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Please place 'perishable_goods_management.csv' in the project directory."
        )
    df = pd.read_csv(path)

    required = ["product_name", "base_price", "days_until_expiry",
                 "initial_quantity", "daily_demand"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    # Drop rows where key fields are NaN
    df = df.dropna(subset=required)
    return df


def train(timesteps: int = TIMESTEPS) -> DQN:
    """Train a DQN agent on the perishable pricing environment."""
    print(f"📦  Loading dataset from {DATA_PATH} ...")
    df = load_data()
    print(f"✅  Loaded {len(df):,} rows.")

    env = Monitor(PerishablePricingEnv(df))

    print(f"🧠  Training DQN for {timesteps:,} timesteps ...")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=500,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
    )
    model.learn(total_timesteps=timesteps)

    model.save(MODEL_PATH)
    print(f"💾  Model saved to {MODEL_PATH}.zip")

    env.close()
    return model


def predict_discount(state: np.ndarray, model_path: str = MODEL_PATH) -> tuple[int, float]:
    """
    Load the saved DQN model and predict the optimal discount for a given state.

    Parameters
    ----------
    state : np.ndarray
        Normalized observation vector [days_until_expiry, stock, daily_demand].
    model_path : str
        Path to the saved model (without .zip extension).

    Returns
    -------
    tuple[int, float]
        (action_index, discount_percentage)  e.g. (3, 0.50)
    """
    full_path = model_path if model_path.endswith(".zip") else model_path + ".zip"
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Trained model not found at {full_path}. Run `python train_agent.py` first."
        )

    model = DQN.load(model_path)
    action, _ = model.predict(state, deterministic=True)
    action_idx = int(action)
    return action_idx, DISCOUNT_MAP[action_idx]


# ---------------------
# CLI entry point
# ---------------------
if __name__ == "__main__":
    trained_model = train()

    # Quick smoke test
    test_state = np.array([0.1, 0.5, 0.3], dtype=np.float32)
    action_idx, discount = predict_discount(test_state)
    print(f"\n🔍  Smoke test — state={test_state}")
    print(f"    Predicted action: {action_idx}  →  {discount*100:.0f}% discount")
