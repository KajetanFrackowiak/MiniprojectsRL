import typing as tt
import gymnasium as gym
import wandb
from collections import defaultdict

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
ALPHA = 0.2
EPISODES = 20

State = int
Action = int
ValuesKey = tt.Tuple[State, Action]


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state, _ = self.env.reset()
        self.values: tt.Dict[ValuesKey] = defaultdict(float)

    def sample_env(self) -> tt.Tuple[State, Action, float, State]:
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, is_tr, _ = self.env.step(action)
        if is_done or is_tr:
            self.state, _ = self.env.reset()
        else:
            self.state = new_state
        return old_state, action, float(reward), new_state

    def best_value_and_action(self, state: State) -> tt.Tuple[float, Action]:
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(
        self, state: State, action: Action, reward: float, next_state: State
    ):
        best_val, _ = self.best_value_and_action(next_state)
        new_val = reward + GAMMA * best_val
        old_val = self.values[(state, action)]
        key = (state, action)
        self.values[key] = old_val * (1 - ALPHA) + new_val * ALPHA

    def play_episode(self, env: gym.Env) -> float:
        total_reward = 0.0
        state, _ = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, is_tr, _ = env.step(action)
            total_reward += reward
            if is_done or is_tr:
                break
            state = new_state
        return total_reward


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()

    wandb.init(project="deep-q-learning", name="q-learning")

    iter_num = 0
    best_reward = 0.0
    while True:
        iter_num += 1
        state, action, reward, next_state = agent.sample_env()
        agent.value_update(state, action, reward, next_state)

        test_reward = 0.0
        for _ in range(EPISODES):
            test_reward += agent.play_episode(test_env)
        test_reward /= EPISODES

        wandb.log({"reward": test_reward, "iteration": iter_num})

        if test_reward > best_reward:
            print(f"{iter_num}: Best reward updated {best_reward:.3f} -> {test_reward}")
            best_reward = test_reward
        if test_reward > 0.80:
            print(f"Solved in {iter_num} iterations!")
            break

    wandb.finish()
