import os
import numpy as np
import gymnasium as gym
from dataclasses import dataclass
import typing as tt
import jax
import jax.numpy as jnp
import optax
import wandb
import haiku as hk
import pickle

HIDDEN_SIZE = 128
BATCH_SIZE = 64
PERCENTILE = 70
INITIAL_EPSILON = 0.1
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
CHECKPOINT_DIR = "checkpoints"

devices = jax.devices()
device = devices[0] if len(devices) > 0 else "cpu"
print(f"Using device: {device}")


# Define the neural network using Haiku
def net_fn(obs_size: int, hidden_size: int, n_actions: int):
    def forward(obs):
        x = jax.nn.relu(hk.Linear(hidden_size)(obs))
        return hk.Linear(n_actions)(x)

    return forward


def create_network(obs_size: int, hidden_size: int, n_actions: int):
    return hk.transform(lambda obs: net_fn(obs_size, hidden_size, n_actions)(obs))


@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int


@dataclass
class Episode:
    reward: float
    steps: tt.List[EpisodeStep]


def iterate_batches(
    env: gym.Env,
    net: hk.Transformed,
    params: hk.Params,
    batch_size: int,
    rng: jax.random.PRNGKey,
    epsilon: float = INITIAL_EPSILON,
) -> tt.Generator[tt.List[Episode], None, None]:
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset()
    while True:
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # Random action
        else:
            obs_v = jnp.array(obs, dtype=jnp.float32)
            act_probs_v = net.apply(params, rng, obs_v[None, :])
            act_probs = np.array(jax.nn.softmax(act_probs_v).squeeze())
            action = np.random.choice(len(act_probs), p=act_probs)  # policy action

        next_obs, reward, is_done, is_trunc, _ = env.step(action)
        episode_reward += float(reward)
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if is_done or is_trunc:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(
    batch: tt.List[Episode], percentile: float
) -> tt.Tuple[jnp.ndarray, jnp.ndarray, float, float]:
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = float(np.percentile(rewards, percentile))
    reward_mean = float(np.mean(rewards))
    train_obs: tt.List[np.ndarray] = []
    train_act: tt.List[int] = []
    for episode in batch:
        if episode.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, episode.steps))
        train_act.extend(map(lambda step: step.action, episode.steps))
    train_obs_v = jnp.vstack(train_obs)
    train_act_v = jnp.array(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


# Updated compute_loss
def compute_loss(logits, labels):
    return optax.softmax_cross_entropy(logits=logits, labels=labels).mean()


def save_checkpoint(params, optimizer_state, filename="checkpoint.pkl"):
    checkpoint = {"params": params, "optimizer_state": optimizer_state}
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(os.path.join(CHECKPOINT_DIR, filename), "wb") as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(filename="checkpoint.pkl"):
    with open(os.path.join(CHECKPOINT_DIR, filename), "rb") as f:
        checkpoint = pickle.load(f)
    return checkpoint["params"], checkpoint["optimizer_state"]


wandb.init(
    project="cartpole-jax",
    config={
        "hidden_size": HIDDEN_SIZE,
        "batch_size": BATCH_SIZE,
        "percentile": PERCENTILE,
    },
)

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    assert env.observation_space.shape is not None
    obs_size = env.observation_space.shape[0]
    assert isinstance(env.action_space, gym.spaces.Discrete)
    n_actions = int(env.action_space.n)

    net = create_network(obs_size, HIDDEN_SIZE, n_actions)
    optimizer = optax.adam(learning_rate=0.01)
    rng = jax.random.PRNGKey(42)
    params = net.init(rng, jnp.ones((1, obs_size)))

    if os.path.exists(os.path.join(CHECKPOINT_DIR, "checkpoint.pkl")):
        print("Loading checkpoint...")
        params, optimizer_state = load_checkpoint("checkpoint.pkl")
    else:
        optimizer_state = optimizer.init(params)

    # Updated loss_fn
    @jax.jit
    def loss_fn(params, obs_v, acts_v):
        logits = net.apply(params, None, obs_v)
        one_hot_labels = jax.nn.one_hot(acts_v, n_actions)
        return compute_loss(logits, one_hot_labels)

    @jax.jit
    def update_fn(params, grads, optimizer_state):
        updates, new_optimizer_state = optimizer.update(grads, optimizer_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_optimizer_state

    # Training loop
    epsilon = INITIAL_EPSILON
    for iter_no, batch in enumerate(
        iterate_batches(env, net, params, BATCH_SIZE, rng, epsilon)
    ):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)

        obs_v = jax.device_put(obs_v, device)
        acts_v = jax.device_put(acts_v, device)

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(params, obs_v, acts_v)

        params, optimizer_state = update_fn(params, grads, optimizer_state)

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        if iter_no % 1000 == 0:
            save_checkpoint(params, optimizer_state, filename="checkpoint.pkl")
            print(f"Checkpoint saved at iteration {iter_no}")

        wandb.log({"loss": loss, "reward_bound": reward_b, "reward_mean": reward_m})
        print(
            f"{iter_no}: loss={loss:.3f}, reward_mean={reward_m:.1f}, rw_bound={reward_b:.1f}"
        )
        if reward_m > 199:
            print("Solved!")
            save_checkpoint(params, optimizer_state, filename="checkpoint.pkl")
            print(f"Checkpoint saved at iteration {iter_no}")
            break
    wandb.finish()
