#!/usr/bin/env python3
import gymnasium as gym
import ale_py
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import wandb
import typing as tt
import dataclasses
import pickle
import os
from datetime import datetime

SEED = 123

@dataclasses.dataclass
class Hyperparams:
    env_name: str
    stop_reward: float
    run_name: str
    replay_size: int
    replay_initial: int
    target_net_sync: int
    epsilon_frames: int
    update_freq: int  # How often to perform a learning update
    frame_skip: int   # Number of frames to skip (action repeat)
    learning_rate: float = 0.00025
    batch_size: int = 32
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_final: float = 0.1
    episodes_to_solve: int = 500

# DeepMind DQN paper parameters
GAME_PARAMS = {
    "pong": Hyperparams(
        env_name="PongNoFrameskip-v4",
        stop_reward=18.0,
        run_name="pong",
        replay_size=1_000_000,      # 1M capacity as in the paper
        replay_initial=50_000,      # 50K initial exploration frames
        target_net_sync=10_000,     # Update target every 10K frames
        epsilon_frames=1_000_000,   # Linear annealing over 1M frames
        update_freq=4,              # Update network every 4 frames
        frame_skip=4,               # Action repeat 4 times
        learning_rate=0.00025,      # RMSProp with 0.00025 learning rate
        gamma=0.99,
        episodes_to_solve=500,
        epsilon_final=0.1,          # Final exploration rate 0.1
    )
}

class DQNNetwork(hk.Module):
    def __init__(self, n_actions, name=None):
        super().__init__(name=name)
        self.n_actions = n_actions

    def __call__(self, x):
        # Normalize between 0 and 1 (as in DeepMind paper)
        x = x.astype(jnp.float32) / 255.0
        
        # Architecture from the Nature paper
        x = hk.Conv2D(32, kernel_shape=8, stride=4, padding="VALID")(x)
        x = jax.nn.relu(x)
        x = hk.Conv2D(64, kernel_shape=4, stride=2, padding="VALID")(x)
        x = jax.nn.relu(x)
        x = hk.Conv2D(64, kernel_shape=3, stride=1, padding="VALID")(x)
        x = jax.nn.relu(x)
        x = hk.Flatten()(x)
        x = hk.Linear(512)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self.n_actions)(x)
        return x

class EpsilonTracker:
    def __init__(self, params: Hyperparams):
        self.params = params
        self.epsilon_slope = (self.params.epsilon_final - self.params.epsilon_start) / self.params.epsilon_frames

    def get_epsilon(self, frame_idx: int) -> float:
        # Linear annealing
        if frame_idx < self.params.epsilon_frames:
            return self.params.epsilon_start + frame_idx * self.epsilon_slope
        else:
            return self.params.epsilon_final

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in batch]

    def __len__(self):
        return len(self.buffer)

def create_dqn_network(obs_shape, n_actions):
    def network(x):
        model = DQNNetwork(n_actions)
        return model(x)

    return hk.without_apply_rng(hk.transform(network))

def loss_fn(params, target_params, network, target_network, batch, gamma):
    states, actions, rewards, next_states, dones = batch

    states = states.astype(jnp.float32)
    actions = actions.astype(jnp.int32)
    rewards = rewards.astype(jnp.float32)
    next_states = next_states.astype(jnp.float32)
    dones = dones.astype(jnp.float32)

    # Compute Q-values for the current states
    q_values = network.apply(params, states)

    # Compute Q-values for the next states using the target network
    next_q_values = target_network.apply(target_params, next_states)

    # Select max action value from target network (DQN approach)
    next_q_value = jnp.max(next_q_values, axis=1)

    # Compute Bellman targets
    targets = rewards + gamma * next_q_value * (1.0 - dones)

    # Select Q-values for the chosen actions
    batch_indices = jnp.arange(q_values.shape[0])
    selected_q_values = q_values[batch_indices, actions]

    # Huber loss as used in the DeepMind paper
    delta = 1.0
    abs_diff = jnp.abs(selected_q_values - targets)
    quadratic = jnp.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return jnp.mean(loss)

def load_latest_checkpoint(checkpoint_dir: str) -> tt.Optional[tt.Tuple[tt.Dict, int, int, list, str]]:
    if not os.path.exists(checkpoint_dir):
        print(f"Directory {checkpoint_dir} does not exist.")
        return None

    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pkl")
    ]

    if not checkpoint_files:
        print("No checkpoint files found.")
        return None

    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

    print(f"Loading checkpoint: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)

    if isinstance(checkpoint_data, dict):
        if "params" in checkpoint_data and "opt_state" in checkpoint_data and "total_frames" in checkpoint_data and "episodes_completed" in checkpoint_data and "episode_rewards" in checkpoint_data:
            params = checkpoint_data["params"]
            opt_state = checkpoint_data["opt_state"]
            total_frames = checkpoint_data["total_frames"]
            episodes_completed = checkpoint_data["episodes_completed"]
            episode_rewards = checkpoint_data["episode_rewards"]
            run_id = checkpoint_data.get("run_id", f"dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            return params, opt_state, total_frames, episodes_completed, episode_rewards, run_id
    
    print("Invalid checkpoint format.")
    return None

# Custom wrapper for reward clipping
class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, reward):
        """Clips rewards to {-1, 0, +1} as per the DeepMind paper"""
        return np.sign(reward)

def make_env(env_name, frame_skip=4):
    """Create Atari environment with proper preprocessing as in the DQN paper"""
    env = gym.make(env_name, render_mode=None)
    
    # Skip the "NoFrameskip" part when applying AtariPreprocessing
    # since we explicitly set frame_skip
    env = ClipRewardEnv(env)  # Reward clipping
    env = GrayscaleObservation(env, keep_dim=False)  # Convert to grayscale
    env = ResizeObservation(env, (84, 84))  # Resize to 84x84
    env = FrameStackObservation(env, 4)  # Stack 4 frames
    return env

def train(params: Hyperparams) -> tt.Optional[int]:
    env = make_env(params.env_name, params.frame_skip)
    env.reset(seed=SEED)
    
    key = jax.random.PRNGKey(SEED)
    network = create_dqn_network((4, 84, 84), env.action_space.n)
    key, subkey = jax.random.split(key)

    # RMSProp optimizer as used in the Nature paper with proper hyperparameters
    optimizer = optax.rmsprop(
        learning_rate=params.learning_rate,
        decay=0.95,
        eps=0.01,
        initial_scale=1.0
    )

    checkpoint_dir = "/content/drive/MyDrive/dqn_checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_data = load_latest_checkpoint(checkpoint_dir)
    if checkpoint_data is not None:
        initial_params, opt_state, total_frames, episodes_completed, episode_rewards, run_id = checkpoint_data
        print(f"Loaded checkpoint with {total_frames} frames completed")
        print(f"Resuming W&B run with run_id: {run_id}")
    else:
        # Initialize parameters with correct input shape (batch, channels, height, width)
        dummy_input = jnp.zeros((1, 4, 84, 84), dtype=jnp.float32)
        initial_params = network.init(subkey, dummy_input)
        total_frames = 0
        episodes_completed = 0
        episode_rewards = []
        run_id = f"dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print("No checkpoint found, starting from scratch")

    wandb.init(
        project="jax-dqn", 
        name="DQN-Nature", 
        id=run_id, 
        resume="allow",
        config=dataclasses.asdict(params)
    )

    target_params = initial_params
    
    opt_state = optimizer.init(initial_params)

    epsilon_tracker = EpsilonTracker(params)
    replay_buffer = ReplayBuffer(params.replay_size)

    @jax.jit
    def update_step(params, target_params, opt_state, batch, gamma):
        grad_fn = jax.value_and_grad(loss_fn, argnums=0)
        loss, grads = grad_fn(params, target_params, network, network, batch, gamma)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state

    frames_since_last_update = 0
    best_mean_reward = float('-inf')
    
    while episodes_completed < params.episodes_to_solve:
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0

        while not done:
            # Convert state from (frames, height, width) to (frames, height, width)
            # JAX expects channel-first format
            state_array = np.array(state)
            
            epsilon = epsilon_tracker.get_epsilon(total_frames)
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                # Add batch dimension
                state_input = state_array[None, ...]
                q_vals = network.apply(initial_params, state_input)
                action = int(jnp.argmax(q_vals[0]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state_array, action, reward, np.array(next_state), float(done))
            state = next_state
            episode_reward += reward
            total_frames += 1
            episode_steps += 1
            frames_since_last_update += 1

            # Only update the network every update_freq frames (4 in the paper)
            if frames_since_last_update >= params.update_freq and len(replay_buffer) >= params.replay_initial:
                frames_since_last_update = 0
                batch = replay_buffer.sample(params.batch_size)
                batch = [jnp.array(x) for x in zip(*batch)]

                loss, initial_params, opt_state = update_step(
                    initial_params, target_params, opt_state, batch, params.gamma
                )

                # Update target network at regular intervals
                if total_frames % params.target_net_sync == 0:
                    target_params = initial_params
                    print(f"Updated target network at frame {total_frames}")

                wandb.log(
                    {
                        "loss": float(loss),
                        "epsilon": epsilon,
                        "total_frames": total_frames
                    },
                    step=total_frames
                )

            # Checkpoint saving logic
            if total_frames % 10_000 == 0:
                checkpoint_path = f"{checkpoint_dir}/checkpoint_{total_frames}.pkl"
                checkpoint_data = {
                    "params": initial_params,
                    "opt_state": opt_state,
                    "total_frames": total_frames,
                    "episodes_completed": episodes_completed,
                    "episode_rewards": episode_rewards,
                    "run_id": run_id
                }
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(checkpoint_data, f)
                print(f"Checkpoint saved at frame {total_frames}")
                
                # Calculate mean reward over last 100 episodes
                if len(episode_rewards) > 100:
                    mean_reward = np.mean(episode_rewards[-100:])
                    if mean_reward > best_mean_reward:
                        best_mean_reward = mean_reward
                        best_model_path = f"{checkpoint_dir}/best_model.pkl"
                        with open(best_model_path, "wb") as f:
                            pickle.dump(checkpoint_data, f)
                        print(f"New best model with mean reward: {mean_reward:.2f}")

            if done:
                wandb.log(
                    {
                        "episode_reward": episode_reward,
                        "episode_steps": episode_steps,
                        "episode": episodes_completed
                    },
                    step=total_frames
                )

                episode_rewards.append(episode_reward)
                episodes_completed += 1
                print(f"Episode {episodes_completed}: Reward = {episode_reward}, Steps: {episode_steps}, Frames: {total_frames}")
                
                # Calculate running average
                if len(episode_rewards) > 100:
                    avg_reward = np.mean(episode_rewards[-100:])
                    print(f"Last 100 episodes average reward: {avg_reward:.2f}")
                    if avg_reward >= params.stop_reward:
                        print(f"Environment solved with average reward {avg_reward:.2f} over 100 episodes!")
                        return episodes_completed

    return None

def setup_jax_gpu():
    print("JAX visible devices:", jax.devices())
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

def main():
    setup_jax_gpu()
    params = GAME_PARAMS["pong"]
    train(params)

if __name__ == "__main__":
    main()
