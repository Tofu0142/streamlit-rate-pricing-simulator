import numpy as np
from typing import Any, Optional, List
import vowpalwabbit
from .environment import BaseEnvironment


class Agent:
    def __init__(
        self,
        env: BaseEnvironment,
        min_action: float,
        max_action: float,
        n_actions: int,
        epsilon: float = 0.1,
        runs: int = 1,
        arg_list: Optional[List] = None,
        extra_arg_list: Optional[List] = None,
        **kwargs,
    ):
        self.env = env
        self.ARG_LIST = None
        self.MIN_ACTION = min_action
        self.MAX_ACTION = max_action
        self.ACTIONS = np.linspace(min_action, max_action, n_actions)
        self.N_ACTIONS = n_actions
        self.EPSILON = epsilon
        self.pass_index = False
        self.history = {}
        self.rewards = []
        self.actions = []
        self.runs = runs
        self.learn = True
        self.logs = []

        if extra_arg_list is None:
            extra_arg_list = []

        if arg_list is None:
            arg_list = [
                f"--cb_explore",
                f"{str(self.N_ACTIONS)}",
                "--epsilon",
                f"{self.EPSILON}",
                "--cb_type",
                "ips",
                "--quiet",
                "-q",
                "::",
            ]

        arg_list = arg_list + extra_arg_list

        self.vw = vowpalwabbit.Workspace(arg_list=arg_list)

    def choose_action(self, context: str, *args, **kwargs):
        raise RuntimeError("Not Implemented!")

    def get_reward(self, action: Any, *args, **kwargs):
        reward = self.env.give_reward(action, *args, **kwargs)
        self.rewards.append(reward)
        return reward

    def update_belief(self, context: str, action: Any, prob: float, reward: float):
        raise RuntimeError("Not Implemented")

    def step(self, *args, **kwargs):
        context = self.env.get_context()
        action, idx, prob = self.choose_action(self.env.format_function(context))
        reward = self.get_reward(action, *args, **kwargs)
        if self.learn:
            if self.pass_index:
                action = idx
            self.update_belief(context, action, prob, reward)

        self.env.step()

    def populate_history(self):
        self.history["rewards"] = self.rewards
        self.history["actions"] = self.actions

    def reset(self):
        self.history = {}
        self.rewards = []
        self.actions = []
        self.prices = []

    def run(self, *args, **kwargs):
        self.reset()
        for n in range(self.runs):
            self.env.reset()
            while self.env.current_index < self.env.n_examples:
                self.step(*args, **kwargs)

            self.populate_history()

        return self.history

    def get_cumulative_avg_reward(self):
        return np.cumsum(self.rewards) / range(1, len(self.rewards) + 1)


######## CB AGENT ########
class CBExploreAgent(Agent):
    def __init__(self, learn=True, update_lower_prices=False, **kwargs):
        super().__init__(**kwargs)

        self.pass_index = True
        self.prices = []
        self.learn = learn
        self.update_lower_prices = update_lower_prices

    def choose_action(self, context, **kwargs):
        self.pmf = np.array(self.vw.predict(context))
        self.pmf /= self.pmf.sum()  # Normalise pmf
        idx = np.random.choice(len(self.ACTIONS), p=self.pmf)

        action = self.ACTIONS[idx]
        self.actions.append(action)

        price = self.action_to_price(action=action)
        self.prices.append(price)

        return price, idx, self.pmf[idx]

    def action_to_price(self, action):
        price = self.env.current_row["sell_nightly_rates"] * (1 + action)
        return price

    def get_reward(self, price, *args, **kwargs):
        reward = self.env.give_reward(price, *args, **kwargs)
        if kwargs["append"]:
            self.rewards.append(reward)

        return reward

    def update_belief(self, context, action, prob, reward):
        # VW has 1-count, idx needs to be increased by one
        # Also VW works with cost, nor rewards
        context_and_labels = self.env.format_function(
            context, (action + 1, -reward, prob)
        )

        self.logs.append(context_and_labels)
        self.vw.learn(context_and_labels)

    def step(self, *args, **kwargs):
        context = self.env.get_context()

        action, idx, prob = self.choose_action(self.env.format_function(context))

        if self.update_lower_prices:
            _range = range(idx + 1)
        else:
            _range = range(1)

        append = True
        for i in _range:
            _idx = idx - i
            if self.update_lower_prices:
                action = self.action_to_price(self.ACTIONS[_idx])
            reward = self.get_reward(action, append=append, n_iter=i, *args, **kwargs)
            append = False  # Append only first action and reward
            if self.learn:
                if self.pass_index:
                    action = _idx
                self.update_belief(context, action, self.pmf[_idx], reward)
            if reward == 0:
                # Only evaluate reward for lower prices if current reward > 0
                break

        self.env.step()

    def populate_history(self):
        super().populate_history()
        self.history["prices"] = self.prices
        self.history["agent"] = self.vw


#### RANDOM AGENT ########


class RandomAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pass_index = True
        self.prices = []

    def choose_action(self, context: str, *args, **kwargs):
        idx = np.random.choice(self.N_ACTIONS)
        action = self.ACTIONS[idx]
        pmf = 1 / len(self.ACTIONS)

        self.actions.append(action)

        price = self.action_to_price(action=self.ACTIONS[idx])

        self.prices.append(price)

        return price, idx, pmf

    def action_to_price(self, action):
        price = self.env.current_row["sell_nightly_rates"] * (1 + action)
        return price

    def get_reward(self, price, *args, **kwargs):
        reward = self.env.give_reward(price, *args, **kwargs)
        self.rewards.append(reward)
        return reward

    def update_belief(self, *args, **kwargs):
        pass


#### LINEAR REGRESSION AGENT ########
class LinearRegressionAgent(Agent):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self.pass_index = True

    def choose_action(self, context: str, *args, **kwargs):
        pmf = np.array(self.vw.predict(context))
        pmf /= pmf.sum()  # Normalise pmf
        idx = np.random.choice(len(self.ACTIONS), p=pmf)

        self.actions.append(self.ACTIONS[idx])

        return self.ACTIONS[idx], idx, pmf[idx]

    def update_belief(self, context, action, prob, reward):
        # VW has 1-count, idx needs to be increased by one
        context_and_labels = self.env.format_function(
            context, (action + 1, -reward, prob)
        )
        self.logs.append(context_and_labels)
        self.vw.learn(context_and_labels)

    def populate_history(self):
        super().populate_history()
        self.history["agent"] = self.vw


class LinearRegressionAgentCATS(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pass_index = True

    def choose_action(self, context: str, *args, **kwargs):
        action, pdf = np.array(self.vw.predict(context))

        self.actions.append(action)

        return action, pdf

    def update_belief(self, context, action, prob, reward):
        # VW has 1-count, idx needs to be increased by one
        context_and_labels = self.env.format_function(
            context, (action + 1, -reward, prob)
        )
        self.logs.append(context_and_labels)
        self.vw.learn(context_and_labels)

    def step(self, *args, **kwargs):
        context = self.env.get_context()
        action, prob = self.choose_action(self.env.format_function(context))
        reward = self.get_reward(action, *args, **kwargs)  # VW thinks in terms of cost
        if self.learn:
            self.update_belief(context, action, prob, reward)

        self.env.step()

    def populate_history(self):
        super().populate_history()
        self.history["agent"] = self.vw
