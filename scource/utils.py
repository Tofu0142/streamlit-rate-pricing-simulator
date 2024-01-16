import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import streamlit as st
import gym

from itertools import count
import random
import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objs as go

from simulator_envs_gym import SimulatorEnv1, LogisticModelFeature
from mab import SimulatorBanditModel

BASE_PRICE = LogisticModelFeature(
    name="base_price",
    model_coefficient=-1 / 25,
    space="simulator1_price",
    mean_base_price=100,
)
APW = LogisticModelFeature(
    name="apw",
    model_coefficient=-1 / 10,
    space="negative_binomial",
    mean=14,
)
OCCOUPANCY = LogisticModelFeature(
    name="occupancy",
    model_coefficient=1 / 10,
    space="beta_distribution",
    alpha=1.5,
    beta=5,
)

NIGHT = LogisticModelFeature(
    name="night",
    model_coefficient=0.1,
    space="random",
    low=1,
    high=8,
)

STARS = LogisticModelFeature(
    name="hotel_level",
    model_coefficient=0.1,
    space="random",
    low=0,
    high=5,
)

FEATURES = {
    "base_price": BASE_PRICE,
    "apw": APW,
    "occupancy": OCCOUPANCY,
    "night": NIGHT,
    "hotel_level": STARS,
}


def get_data(features_choice, arms_range, n_arm, n_data):
    filtered_dict = {key: FEATURES[key] for key in features_choice}
    env = SimulatorEnv1(
        discrete_action_space=True,
        n_actions=n_arm,
        intercept=10.0,
        features=filtered_dict,
        n_reps=n_data,
        first_arm=arms_range[0],
        last_arm=arms_range[1],
    )

    terminated = False
    action = env.action_space.sample()[0]
    training_batch_size = n_data
    data = []
    env.reset()
    observation = env._get_obs()
    
    while not terminated:
        if env.current_index < training_batch_size:
            action = env.action_space.sample()[0]
        else:
            terminated = True
        new_observation, reward, terminated, truncated, info = env.step(action)
        data.append([observation, action, reward])
        observation = new_observation

    df = pd.DataFrame(data, columns=["observation", "action", "reward"])
    observations_df = df["observation"].apply(pd.Series)
    df = pd.concat([observations_df, df.drop("observation", axis=1)], axis=1)

    return df, env

def train_and_eva(env, model, train, test):

    model.fit(
        decisions=train["action"],
        rewards=train["reward"],
        contexts=train.drop(["action", "reward"], axis=1),
    )
    predictions = model.predict(test.drop(["action", "reward"], axis=1))
    rewards_ = []
    for i in predictions:
        _, reward, _, _, _ = env.step(i)
        rewards_.append(reward)
    return predictions, rewards_

def train_and_eva1(features_choice, 
                   arms_range, 
                   n_arm, 
                   n_data, 
                   learning_policy
                   ):
    filtered_dict = {key: FEATURES[key] for key in features_choice}
    env = SimulatorEnv1(
        discrete_action_space=True,
        n_actions=n_arm,
        intercept=10.0,
        features=filtered_dict,
        n_reps=n_data,
        first_arm=arms_range[0],
        last_arm=arms_range[1],
    )

    terminated = False
    action = env.action_space.sample()[0]
    training_batch_size = 200
    data = []
    env.reset()
    observation = env._get_obs()
    
    model = SimulatorBanditModel(
        first_arm=arms_range[0],
        last_arm=arms_range[1],
        n_arms=n_arm,
        learning_policy=learning_policy,
        
    )
    while not terminated:
        if env.current_index < training_batch_size:
            action = env.action_space.sample()[0]
        else:
            action = model.choose_action()
        new_observation, reward, terminated, truncated, info = env.step(action)
        if "action" in info:
            _action = info["action"]
        else:
            _action = action

        model.collect(observation, _action, reward)
        observation = new_observation

        if (env.current_index >0 ) and (env.current_index % training_batch_size == 0):
            model.train()
    return model 
    
def plot_cumulative_rewards(dfs, title=None, filename='animation.mp4'):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    plt.style.use("ggplot")
    lines = {model_name: df['reward'].cumsum() / np.arange(1, len(df) + 1) for model_name, df in dfs.items()}
    data = {model_name: [] for model_name in lines}
    x_values = []
    myvar = count(0, 3)

    def animate(i):
        x = next(myvar)
        x_values.append(x)
        for model_name, line in lines.items():
            data[model_name].append(line[i] if i < len(line) else line[-1])

        axes.clear()
        for model_name, y_values in data.items():
            axes.plot(x_values, y_values, label=model_name)
        axes.legend()

    anim = FuncAnimation(fig, animate, frames=len(max(lines.values(), key=len)), interval=30)
    anim.save(filename, writer='ffmpeg')
    plt.close(fig)
    return filename

def create_cumulative_rewards_animation(dfs, ind):

    # Initialize a figure
    fig = go.Figure()

    # Add the initial trace for each model
    for model_name, df in dfs.items():
        fig.add_trace(
            go.Scatter(
                x=[0], 
                y=[0], 
                mode='lines',
                name=model_name
            )
        )
    
    # Add frames for each point in the data
    frames = []
    for i in range(1, max(len(df) for df in dfs.values()) + 1, 200):
        frame_data = []
        for model_name, df in dfs.items():
            df_subset = df[:i]
            if ind == 'reward':
                cumulative_sum = df_subset['reward'].cumsum()
                n_observations = np.arange(1, len(df_subset) + 1)
                y_value = cumulative_sum / n_observations
            else:
                ttv = df_subset['reward'] * df_subset['base_price'] * (1 + df_subset['action'])
                n_observations = np.arange(1, len(df_subset) + 1)
                y_value = ttv.cumsum() / n_observations
            frame_data.append(go.Scatter(x=n_observations, y=y_value, mode='lines', name=model_name))
        frames.append(go.Frame(data=frame_data, name=str(i)))

    fig.frames = frames

    # Update layout for animation
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                }
            ]
        }]
    )

    return fig

