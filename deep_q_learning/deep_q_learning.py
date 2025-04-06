#!/usr/bin/env python3
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import ale_py
import os
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import wandb
import numpy as np
import typing as tt
import dataclasses
import pickle
from datetime import timedelta, datetime

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
    learning_rate: float = 0.0001
    batch_size: int = 32
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_final: float = 0.1
    episodes_to_solve: int = 500

GAME_PARAMS = {
    "pong": Hyperparams(
        env_name="PongNoFrameskip-v4",
        stop_reward=18.0,
        run_name="pong",
        replay_size=100_000,
        replay_initial=10_000,
        target_net_sync=10_000, 
        epsilon_frames=1_000_000,
        epsilon_final=0.02,
        learning_rate=9.932831968547505e-05, 
        gamma=0.98,
        episodes_to_solve=340, 
    )
}

class DQNNetwork(hk.Module):
    def __init__(self, n_actions, name=None):
        super().__init__(name=name)
        self.n_actions = n_actions

    def __call__(self, x):
        x = x.astype(jnp.float32) / 255.0
        conv_layers = hk.Sequential(
            [
                hk.Conv2D(32, kernel_shape=8, stride=4),
                jax.nn.relu,
                hk.Conv2D(64, kernel_shape=4, stride=2),
                jax.nn.relu,
                hk.Conv2D(64, kernel_shape=3, stride=1),
                jax.nn.relu,
                hk.Flatten(),
            ]
        )

        conv_out = conv_layers(x)
        fc_layers = hk.Sequential(
            [hk.Linear(512), jax.nn.relu, hk.Linear(self.n_actions)]
        )

        return fc_layers(conv_out)

class EpsilonTracker:
    def __init__(self, params: Hyperparams):
        self.params = params

    def get_epsilon(self, frame_idx: int) -> float:
        eps = self.params.epsilon_start - frame_idx / self.params.epsilon_frames
        return max(self.params.epsilon_final, eps)

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

    # Pure DQN: Select max action value from target network
    next_q_value = jnp.max(next_q_values, axis=1)

    # Compute Bellman targets (equivalent to PyTorch's next_state_vals[done_mask] = 0.0)
    targets = rewards + gamma * next_q_value * (1.0 - dones)

    # Select Q-values for the chosen actions
    batch_indices = jnp.arange(q_values.shape[0])
    selected_q_values = q_values[batch_indices, actions]

    loss = jnp.mean((selected_q_values - targets) ** 2)

    return loss

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

    print(f"Loaded checkpoint keys: {list(checkpoint_data.keys())}")

    if isinstance(checkpoint_data, dict):
        if "params" in checkpoint_data and "total_frames" in checkpoint_data and "episodes_completed" in checkpoint_data and "episode_rewards" in checkpoint_data:
            params = checkpoint_data["params"]
            total_frames = checkpoint_data["total_frames"]
            episodes_completed = checkpoint_data["episodes_completed"]
            episode_rewards = checkpoint_data["episode_rewards"]
            run_id = checkpoint_data.get("run_id", f"pure_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            return params, total_frames, episodes_completed, episode_rewards, run_id
        else:
            print("Checkpoint missing some data.")
            return None

    print("Unexpected checkpoint format.")
    return None

def make_env(env_name):
    env = gym.make(env_name)
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, 4)
    return env

def train(params: Hyperparams) -> tt.Optional[int]:
    env = gym.make(params.env_name)
    env.reset(seed=SEED)

    key = jax.random.PRNGKey(SEED)
    network = create_dqn_network((84, 84, 4), env.action_space.n)
    key, subkey = jax.random.split(key)

    checkpoint_dir = "/content/drive/MyDrive/dqn_checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_data = load_latest_checkpoint(checkpoint_dir)
    if checkpoint_data is not None:
        initial_params, total_frames, episodes_completed, episode_rewards, run_id = checkpoint_data
        print(f"Loaded checkpoint with {len(initial_params)} layers.")
        print(f"Resuming W&B run with run_id: {run_id}")
    else:
        initial_params = network.init(
            subkey, jnp.zeros((1, *env.observation_space.shape), dtype=jnp.float32)
        )
        total_frames = 0
        episodes_completed = 0
        episode_rewards = []
        run_id = f"pure_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print("No checkpoint found, starting from scratch")

    wandb.init(project="jax-dqn", name="Pure DQN", id=run_id, resume="allow", config=dataclasses.asdict(params))

    target_params = initial_params
    optimizer = optax.adam(params.learning_rate)
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

    while episodes_completed < params.episodes_to_solve:
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0

        while not done:
            epsilon = epsilon_tracker.get_epsilon(total_frames)
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_vals = network.apply(initial_params, state[None])
                action = int(jnp.argmax(q_vals))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward
            total_frames += 1
            episode_steps += 1

            if len(replay_buffer.buffer) >= params.replay_initial:
                batch = replay_buffer.sample(params.batch_size)
                batch = [jnp.array(x) for x in zip(*batch)]

                loss, initial_params, opt_state = update_step(
                    initial_params, target_params, opt_state, batch, params.gamma
                )

                if total_frames % params.target_net_sync == 0:
                    target_params = initial_params

                wandb.log(
                    {
                        "loss": float(loss),
                        "epsilon": epsilon,
                        "total_frames": total_frames
                    },
                    step=total_frames
                )

            if total_frames % 10_000 == 0:
                checkpoint_path = f"{checkpoint_dir}/checkpoint_{total_frames}.pkl"
                checkpoint_data = {
                    "params": initial_params,
                    "total_frames": total_frames,
                    "episodes_completed": episodes_completed,
                    "episode_rewards": episode_rewards,
                    "run_id": run_id
                }
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(checkpoint_data, f)
                print(f"Checkpoint {checkpoint_path} saved at frame {total_frames}")

            if done:
                wandb.log(
                    {
                        "episode_reward": episode_reward,
                        "episode_steps": episode_steps
                    },
                    step=total_frames
                )

                episode_rewards.append(episode_reward)
                episodes_completed += 1
                print(f"Episode {episodes_completed}: Reward = {episode_reward}, Steps: {episode_steps}")

                if episodes_completed % 100 == 0:
                    env.render()

                if episode_reward >= params.stop_reward:
                    print(f"Environment solved in {episodes_completed} episodes!")
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
