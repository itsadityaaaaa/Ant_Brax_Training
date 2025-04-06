# train_brax_ant_policy.py

from brax import envs
from brax.training.agents.ppo import train
from brax.training import configuration
import jax
import flax
import os

# Set up the Ant environment
env_name = "ant"
env = envs.get_environment(env_name)

# Define training configuration
config = configuration.TrainingConfig(
    env_name=env_name,
    num_timesteps=1_000_000,
    episode_length=1000,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=32,
    num_epochs=10,
    discounting=0.99,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    seed=42,
)

# Train the PPO agent
inference_fn, params, _ = train(
    environment_fn=lambda **kwargs: env,
    config=config,
    progress_fn=lambda **kwargs: print("Training..."),
)

# Save the policy using flax
save_path = "ppo_ant_policy.msgpack"
bytes_output = flax.serialization.to_bytes(params)
with open(save_path, "wb") as f:
    f.write(bytes_output)

print(f"\n✅ Training complete! Policy saved to '{save_path}'")
