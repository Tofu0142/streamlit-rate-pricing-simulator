"""
This file contains the functions for plotting the results of the simulations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objs as go
import plotly.express as px



def plot_hotel_reward_distribution(prepared_data, selected_action):
    # Combine data from all models for the selected action
    combined_data_list = []
    for model_name, df in prepared_data.items():
        filtered_df = df[df["action"] == selected_action].copy()
        filtered_df["model"] = model_name
        combined_data_list.append(filtered_df)

    combined_data = pd.concat(combined_data_list)
    # Create the plot
    fig = px.bar(
        combined_data, x="hotel_level", y="reward", color="model", barmode="group"
    )
    fig.update_layout(
        title="Average Gross Profit Distribution by Hotel Level and Model",
        xaxis_title="Hotel Level",
        yaxis_title="Avg Gross Profit",
    )
    return fig


def plot_customer_reward_distribution(prepared_data, selected_action):
    # Combine data from all models for the selected action
    combined_data_list = []
    for model_name, df in prepared_data.items():
        filtered_df = df[df["action"] == selected_action].copy()
        filtered_df["model"] = model_name
        combined_data_list.append(filtered_df)

    combined_data = pd.concat(combined_data_list)
    # Create the plot
    fig = px.bar(
        combined_data, x="customer_type", y="reward", color="model", barmode="group"
    )
    fig.update_layout(
        title="Average Gross Profit Distribution by Customer Type and Model",
        xaxis_title="Customer Type",
        yaxis_title="Avg Gross Profit",
    )
    return fig

def plot_cumulative_rewards(dfs, title=None, filename="animation.mp4"):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    plt.style.use("ggplot")
    lines = {
        model_name: df["converted"].cumsum() / np.arange(1, len(df) + 1)
        for model_name, df in dfs.items()
    }
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

    anim = FuncAnimation(
        fig, animate, frames=len(max(lines.values(), key=len)), interval=30
    )
    anim.save(filename, writer="ffmpeg")
    plt.close(fig)
    return filename


def create_cumulative_rewards_animation(dfs, ind):
    # Initialize a figure
    fig = go.Figure()

    # Add the initial trace for each model
    for model_name, df in dfs.items():
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="lines", name=model_name))

    # Add frames for each point in the data
    frames = []
    for i in range(1, max(len(df) for df in dfs.values()) + 1, 200):
        frame_data = []
        for model_name, df in dfs.items():
            df_subset = df[:i]
            if ind == "converted":
                cumulative_sum = df_subset["converted"].cumsum()
                n_observations = np.arange(1, len(df_subset) + 1)
                y_value = cumulative_sum / n_observations
            else:
                ttv = (
                    df_subset["reward"]
                   
                )
                n_observations = np.arange(1, len(df_subset) + 1)
                y_value = ttv.cumsum() / n_observations
            frame_data.append(
                go.Scatter(x=n_observations, y=y_value, mode="lines", name=model_name)
            )
        frames.append(go.Frame(data=frame_data, name=str(i)))

    fig.frames = frames
    shapes = []
    for x in range(2000,  max(len(df) for df in dfs.values()) + 1, 2000):
        shapes.append({
            'type': 'line',
            'x0': x,
            'y0': 0,
            'x1': x,
            'y1': 1,
            'xref': 'x',
            'yref': 'paper',
            'line': {'color': 'red', 'width': 1}
        })
    # Update layout for animation
    fig.update_layout(
        shapes=shapes,
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ]
    )

    return fig


def create_cumulative_rewards_animation_5(dfs, ind):
    # Initialize a figure
    fig = go.Figure()

    # Add the initial trace for each model
    for model_name, df in dfs.items():
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="lines", name=model_name))

    # Add frames for each point in the data
    frames = []
    for i in range(1, max(len(df) for df in dfs.values()) + 1, 200):
        frame_data = []
        for model_name, df in dfs.items():
            df_subset = df.iloc[:i]
            if ind == "converted":
                cumulative_sum = df_subset["converted"].cumsum()
                n_observations = np.arange(1, len(df_subset) + 1)
                y_value = cumulative_sum / n_observations
            else:
                ttv = df_subset["reward"].cumsum()
                n_observations = np.arange(1, len(df_subset) + 1)
                y_value = ttv / n_observations
            frame_data.append(
                go.Scatter(x=n_observations, y=y_value, mode="lines", name=model_name)
            )
        frames.append(go.Frame(data=frame_data, name=str(i)))

    fig.frames = frames

    shapes = []
    for x in range(2000, max(len(df) for df in dfs.values()) + 1, 2000):
        shapes.append({
            'type': 'line',
            'x0': x,
            'y0': 0,
            'x1': x,
            'y1': 1,
            'xref': 'x',
            'yref': 'paper',
            'line': {'color': 'red', 'width': 1}
        })

    # Add additional lines or markers for promotion and demand change milestones
    promotion_marks = [6000, 12000, 18000]  # Every 6k for promotion changes
    demand_changes = [8000, 16000]  # Specific points for demand changes

    for milestone in promotion_marks:
        shapes.append({
            'type': 'line',
            'x0': milestone,
            'y0': 0,
            'x1': milestone,
            'y1': 1,
            'xref': 'x',
            'yref': 'paper',
            'line': {'color': 'green', 'width': 2, 'dash': 'dot'}
        })

    for demand_change in demand_changes:
        shapes.append({
            'type': 'line',
            'x0': demand_change,
            'y0': 0,
            'x1': demand_change,
            'y1': 1,
            'xref': 'x',
            'yref': 'paper',
            'line': {'color': 'blue', 'width': 2, 'dash': 'dash'}
        })

    # Adding annotations for clarity
    annotations = []
    for milestone in promotion_marks:
        annotations.append({
            'x': milestone,
            'y': 1,
            'xref': 'x',
            'yref': 'paper',
            'text': 'Promotion Change',
            'showarrow': True,
            'arrowhead': 7,
            'ax': 0,
            'ay': -40
        })

    for demand_change in demand_changes:
        annotations.append({
            'x': demand_change,
            'y': 1,
            'xref': 'x',
            'yref': 'paper',
            'text': 'Demand Change',
            'showarrow': True,
            'arrowhead': 7,
            'ax': 0,
            'ay': -40
        })


    # Define annotations for specific events
    annotations = [
        {"x": x, "y": 1.05, "xref": "x", "yref": "paper", "text": "Promotion Change", "showarrow": False, "font": {"color": "green"}} for x in promotion_marks
    ] + [
        {"x": x, "y": 1.05, "xref": "x", "yref": "paper", "text": "Demand Change", "showarrow": False, "font": {"color": "blue"}} for x in demand_changes
    ]

    # Update layout for animation and add shapes and annotations
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                ],
            }
        ]
    )

    return fig




def create_cumulative_rewards_animation_6(dfs, ind):
    # Initialize a figure
    fig = go.Figure()

    # Add the initial trace for each model
    for model_name, df in dfs.items():
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="lines", name=model_name))

    # Add frames for each point in the data
    frames = []
    for i in range(1, max(len(df) for df in dfs.values()) + 1, 200):
        frame_data = []
        for model_name, df in dfs.items():
            df_subset = df.iloc[:i]
            if ind == "converted":
                cumulative_sum = df_subset["converted"].cumsum()
                n_observations = np.arange(1, len(df_subset) + 1)
                y_value = cumulative_sum / n_observations
            else:
                ttv = df_subset["reward"].cumsum()
                n_observations = np.arange(1, len(df_subset) + 1)
                y_value = ttv / n_observations
            frame_data.append(
                go.Scatter(x=n_observations, y=y_value, mode="lines", name=model_name)
            )
        frames.append(go.Frame(data=frame_data, name=str(i)))

    fig.frames = frames

    shapes = []
    for x in range(2000, max(len(df) for df in dfs.values()) + 1, 2000):
        shapes.append({
            'type': 'line',
            'x0': x,
            'y0': 0,
            'x1': x,
            'y1': 1,
            'xref': 'x',
            'yref': 'paper',
            'line': {'color': 'red', 'width': 1}
        })

    # Add additional lines or markers for promotion and demand change milestones
    promotion_marks = [6000, 12000, 18000]  # Every 6k for promotion changes
    demand_changes = [8000, 14000, 20000]  # Specific points for demand changes

    for milestone in promotion_marks:
        shapes.append({
            'type': 'line',
            'x0': milestone,
            'y0': 0,
            'x1': milestone,
            'y1': 1,
            'xref': 'x',
            'yref': 'paper',
            'line': {'color': 'green', 'width': 2, 'dash': 'dot'}
        })

    for demand_change in demand_changes:
        shapes.append({
            'type': 'line',
            'x0': demand_change,
            'y0': 0,
            'x1': demand_change,
            'y1': 1,
            'xref': 'x',
            'yref': 'paper',
            'line': {'color': 'blue', 'width': 2, 'dash': 'dash'}
        })

    # Adding annotations for clarity
    annotations = []
    for milestone in promotion_marks:
        annotations.append({
            'x': milestone,
            'y': 1,
            'xref': 'x',
            'yref': 'paper',
            'text': 'Promotion Change',
            'showarrow': True,
            'arrowhead': 7,
            'ax': 0,
            'ay': -40
        })

    for demand_change in demand_changes:
        annotations.append({
            'x': demand_change,
            'y': 1,
            'xref': 'x',
            'yref': 'paper',
            'text': 'Demand Change',
            'showarrow': True,
            'arrowhead': 7,
            'ax': 0,
            'ay': -40
        })


    # Define annotations for specific events
    annotations = [
        {"x": x, "y": 1.05, "xref": "x", "yref": "paper", "text": "Promotion Change", "showarrow": False, "font": {"color": "green"}} for x in promotion_marks
    ] + [
        {"x": x, "y": 1.05, "xref": "x", "yref": "paper", "text": "Demand Change", "showarrow": False, "font": {"color": "blue"}} for x in demand_changes
    ]

    # Update layout for animation and add shapes and annotations
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                ],
            }
        ]
    )

    return fig