# test_brax_ant_policy.py

import flax
import jax
import numpy as np
from brax import envs
from brax.training.agents.ppo import inference as ppo_inference

# Load the trained policy
env = envs.get_environment("ant")
params_path = "ppo_ant_policy.msgpack"

with open(params_path, "rb") as f:
    params = flax.serialization.from_bytes(None, f.read())

# Create inference function
inference_fn = ppo_inference.make_inference_fn(env.observation_size, env.action_size)
policy = inference_fn(params)

# Simulate one episode
state = env.reset(rng=jax.random.PRNGKey(0))
total_reward = 0

for step in range(1000):
    action = policy(state.obs)
    state = env.step(state, action)
    total_reward += state.reward
    if state.done:
        break

print(f"\n🏁 Test completed! Total episode reward: {total_reward:.2f}")
