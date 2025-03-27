import os
import pickle
import gymnasium as gym
import gymnasium.spaces
import gymnasium.wrappers
import gymnasium.vector
import gymnasium.envs.toy_text.frozen_lake
from dataclasses import dataclass
import numpy as np
import typing as tt
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import wandb

BATCH_SIZE = 100
NUM_ENVS = 16
CHECKPOINT_DIR = "checkpoints/frozenlake"
GAMMA = 0.9
PERCENTILE = 30

devices = jax.devices()
device = devices[0] if len(devices) > 0 else "cpu"
print(f"Using device: {device}")


# Define the neural network using Haiku
def net_fn(obs_size: int, n_actions: int):
    def forward(obs):
        x = jax.nn.relu(hk.Linear(128)(obs))
        return hk.Linear(n_actions)(x)

    return forward


def create_network(obs_size: int, n_actions: int):
    return hk.transform(lambda obs: net_fn(obs_size, n_actions)(obs))


@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int


@dataclass
class Episode:
    reward: float
    steps: tt.List[EpisodeStep]


class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, (env.observation_space.n,), dtype=np.float32
        )

    def observation(self, observation):
        res = jnp.zeros(self.observation_space.shape, dtype=jnp.float32)
        res = res.at[observation].set(1.0)
        return res


def vectorized_iterate_batches(
    env: gym.vector.VectorEnv,
    net: hk.Transformed,
    params: hk.Params,
    batch_size: int,
    rng: jax.random.PRNGKey,
) -> tt.Generator[tt.List[Episode], None, None]:
    batch = []
    episode_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
    episode_steps = [[] for _ in range(NUM_ENVS)]
    obs, _ = env.reset()
    rngs = jax.random.split(rng, NUM_ENVS)
    while True:
        obs_v = jnp.array(obs, dtype=jnp.float32)
        act_probs_v = net.apply(params, None, obs_v)
        act_probs = jax.nn.softmax(act_probs_v)

        # Sample actions for all envs at once
        actions = jax.vmap(lambda p, r: jax.random.choice(r, len(p), p=p))(
            act_probs, rngs
        )
        actions = np.array(actions)  # TO CPU for env.step
        rngs = jax.random.split(rng, NUM_ENVS)

        next_obs, rewards, dones, truncs, _ = env.step(actions)
        rewards = rewards.astype(np.float32)

        for i in range(NUM_ENVS):
            step = EpisodeStep(observation=obs[i], action=actions[i])
            episode_steps[i].append(step)
            episode_rewards[i] += rewards[i]

            if dones[i] or truncs[i]:
                e = Episode(reward=episode_rewards[i], steps=episode_steps[i])
                batch.append(e)
                episode_rewards[i] = 0.0
                episode_steps[i] = []

        if len(batch) >= batch_size:
            yield batch[:batch_size]
            batch = batch[batch_size:]
        obs = next_obs


def filter_fun(s):
    return s.reward * (GAMMA ** len(s.steps))


def filter_batch(
    batch: tt.List[Episode], percentile: float
) -> tt.Tuple[jnp.ndarray, jnp.ndarray, float, float]:
    disc_rewards = list(map(filter_fun, batch))
    reward_bound = float(np.percentile(disc_rewards, percentile))

    train_obs = []
    train_act = []
    elite_batch = []
    for example, discounted_reward in zip(batch, disc_rewards):
        if discounted_reward >= reward_bound:
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))
            elite_batch.append(example)

    return elite_batch, train_obs, train_act, reward_bound


def compute_loss(logits, labels):
    return optax.softmax_cross_entropy(logits=logits, labels=labels)


def save_checkpoint(params, optimizer_state, filename="checkpoint.pkl"):
    checkpoint = {"params": params, "optimizer_state": optimizer_state}
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(os.path.join(CHECKPOINT_DIR, filename), "wb") as file:
        pickle.dump(checkpoint, file)


wandb.init(
    project="cross_entropy",
    name="frozenlake",
    config={"batch_size": BATCH_SIZE, "percentile": PERCENTILE},
)
if __name__ == "__main__":
    # is_slippery=False to have deterministic environment
    env = gym.vector.SyncVectorEnv(
        [
            lambda: DiscreteOneHotWrapper(gym.make("FrozenLake-v1", is_slippery=False))
            for _ in range(NUM_ENVS)
        ]
    )
    obs_size = env.single_observation_space.shape[0]
    n_actions = env.single_action_space.n

    net = create_network(obs_size, n_actions)
    optimizer = optax.adam(learning_rate=0.01)
    rng = jax.random.PRNGKey(42)
    params = net.init(rng, jnp.ones((1, obs_size)))
    optimizer_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, obs_v, acts_v):
        logits = net.apply(params, None, obs_v)
        one_hot_labels = jax.nn.one_hot(acts_v, n_actions)
        return compute_loss(logits, one_hot_labels).mean()

    @jax.jit
    def update_fn(params, grads, optimizer_state):
        updates, new_optimizer_state = optimizer.update(grads, optimizer_state)
        return optax.apply_updates(params, updates), new_optimizer_state

    full_batch = []
    for iter_no, batch in enumerate(
        vectorized_iterate_batches(env, net, params, BATCH_SIZE, rng)
    ):
        reward_mean = float(np.mean([e.reward for e in batch]))
        full_batch, obs_v, acts_v, reward_b = filter_batch(
            full_batch + batch, PERCENTILE
        )
        if not full_batch:
            print(f"I have to continue, not full_batch. Iter: {iter_no}")
            continue

        obs_v = jnp.array(obs_v, dtype=jnp.float32)
        acts_v = jnp.array(acts_v, dtype=jnp.int32)
        obs_v, acts_v = jax.device_put(obs_v, device), jax.device_put(acts_v, device)
        full_batch = full_batch[-500:]

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(params, obs_v, acts_v)
        params, optimizer_state = update_fn(
            params, grads, optimizer_state
        )

        if iter_no % 100 == 0:
            wandb.log(
                {
                    "loss": float(loss),
                    "reward_bound": reward_b,
                    "reward_mean": reward_mean,
                }
            )
            print(
                f"{iter_no}: loss={loss:.3f}, reward_mean={reward_mean:.1f}, reward_bound={reward_b:.3f}"
            )
            save_checkpoint(
                params, optimizer_state, filename=f"checkpoint{iter_no}.pkl"
            )
            print(f"Checkpoint saved at iteration {iter_no}")

        if reward_mean > 0.8:
            print("Solved!")
            save_checkpoint(
                params, optimizer_state, filename=f"checkpoint{iter_no}.pkl"
            )
            break
    wandb.finish()
