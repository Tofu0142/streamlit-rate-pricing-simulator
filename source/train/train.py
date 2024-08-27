import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import mlflow

import pandas as pd
import matplotlib.pyplot as plt
from celluloid import Camera

from train.train_utils import *
from mab import SimulatorBanditModel
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy




FEATURES = {
    "base_price": BASE_PRICE,
    "apw": APW,
    "occupancy": OCCOUPANCY,
    "night": NIGHT,
    "hotel_level": STARS,
    "promotion": PROMOTION,
    "customer_type": CUSTOMER_TYPE,
}


MODELS = {
    "Random": LearningPolicy.Random(),
    "UCB1": LearningPolicy.UCB1(alpha=1.2),
    "EG": LearningPolicy.EpsilonGreedy(epsilon=0.2),
    "TS": LearningPolicy.ThompsonSampling(),
}



# this function is used for scenario 1-4
def train_model(
    exp_name, n_data, arms_range, n_arm, features_choice, models_choice, retrain_mode, scenario
):
    experiment = mlflow.set_experiment(exp_name)
    results = {}

    train_data, train_env = get_data(features_choice, arms_range, n_arm, n_data, scenario)

    for model in models_choice:
        if model != 'Fixed':
            with mlflow.start_run(experiment_id=experiment.experiment_id):
                mlflow.log_param("features", features_choice)
                mlflow.log_param("arms", arms_range)
                mlflow.log_param("n_arm", n_arm)
                mlflow.log_param("n_data", n_data)
                mlflow.log_param("model", model)
                # print( MODELS[model])
                #print("before", MODELS[model], train_env.current_index)
                train_env.current_index = 4000
                #print("after", MODELS[model], train_env.current_index)
            
                agent = train_profit_reward_1(
                    train_data.copy(),
                    train_env,
                    retrain_mode,
                    arms_range,
                    n_arm,
                    MODELS[model],
                )

                conversation_rate = agent.data.converted.mean().round(2)
                mlflow.log_metric("conversation_rate", conversation_rate)
                mlflow.pyfunc.log_model(
                    python_model=agent, registered_model_name=model, artifact_path="mlruns"
                )

                profit = (
                   agent.data.reward
                    .mean()
                    .round(2)
                )
                mlflow.log_metric("Gross Profit", profit)

                # Calculate cumulative count of actions
                df = agent.data.copy()
                actions = df['action'].unique()

                # Initialize a DataFrame to store cumulative counts
                cumulative_counts = pd.DataFrame(index=df.index, columns=actions).fillna(0)

                # Calculate cumulative count for each action
                for action in actions:
                    cumulative_counts[action] = (df['action'] == action).cumsum()

                # Convert counts to fractions
                cumulative_fractions = cumulative_counts.div(cumulative_counts.sum(axis=1), axis=0)

                fig, ax = plt.subplots(figsize=(10, 6))
                for action in actions:
                    plt.plot(cumulative_fractions.index, cumulative_fractions[action], label=f'Action {action}')

                ax.set_xlabel('Time')
                ax.set_ylabel('Fraction of Action Chosen')
                ax.set_title('Fraction of Each Action Over Time')
                ax.legend()
                mlflow.log_figure(fig, "action_fraction.png")

            results[model] = agent.data
        elif model == "Fixed":
            results[model] = train_and_eva_fixed( train_env)

    return results

# ---------------- train model for scenario 5 ---------------- #
def train_model1(exp_name, arms_range, n_arm, models_choice, retrain_mode, scenario):
    experiment = mlflow.set_experiment(exp_name)
    results = {}
    ini_data, env  = get_data_5(arms_range, n_arm)
    for model in models_choice:
        if model != 'Fixed':
            with mlflow.start_run(experiment_id=experiment.experiment_id):
                mlflow.log_param("arms", arms_range)
                mlflow.log_param("n_arm", n_arm)
                mlflow.log_param("n_data", 20000)
                mlflow.log_param("model", model)
                agent, env = train_profit_reward_5(
                    ini_data.copy(),
                    env,
                    10,
                    retrain_mode,
                    arms_range,
                    n_arm,
                    MODELS[model],
                )

                conversation_rate = agent.data.converted.mean().round(2)
                mlflow.log_metric("conversation_rate", conversation_rate)
                mlflow.pyfunc.log_model(
                    python_model=agent, registered_model_name=model, artifact_path="mlruns"
                )

                Profit_avg = (
                    agent.data.reward 
                    .mean()
                    .round(2)
                )
                mlflow.log_metric("Avg Gross Profit", Profit_avg)

                # Calculate cumulative count of actions
                df = agent.data.copy()
                actions = df['action'].unique()

                # Initialize a DataFrame to store cumulative counts
                cumulative_counts = pd.DataFrame(index=df.index, columns=actions).fillna(0)

                # Calculate cumulative count for each action
                for action in actions:
                    cumulative_counts[action] = (df['action'] == action).cumsum()

                # Convert counts to fractions
                cumulative_fractions = cumulative_counts.div(cumulative_counts.sum(axis=1), axis=0)

                # Plotting
                fig, ax = plt.subplots(figsize=(10, 6))
                for action in actions:
                    plt.plot(cumulative_fractions.index, cumulative_fractions[action], label=f'Action {action}')

                ax.set_xlabel('Time')
                ax.set_ylabel('Fraction of Action Chosen')
                ax.set_title('Fraction of Each Action Over Time')
                ax.legend()
                mlflow.log_figure(fig, "action_fraction.png")
            results[model] = agent.data

    
        df = results['Random']
        results['Fixed'] = train_and_eva_fixed_5( env, df )

    return results

# ---------------- train model for scenario 6 ---------------- #
def train_model_6(exp_name, arms_range, n_arm, models_choice, retrain_mode):
    experiment = mlflow.set_experiment(exp_name)
    results = {}
    ini_data, env  = get_data_6(arms_range, n_arm)
    for model in models_choice:
        if model != 'Fixed':
            with mlflow.start_run(experiment_id=experiment.experiment_id):
                mlflow.log_param("arms", arms_range)
                mlflow.log_param("n_arm", n_arm)
                mlflow.log_param("n_data", 40000)
                mlflow.log_param("model", model)
                agent, env = train_profit_reward_5(
                    ini_data.copy(),
                    env,
                    20,
                    retrain_mode,
                    arms_range,
                    n_arm,
                    MODELS[model],
                )

                conversation_rate = agent.data.converted.mean().round(2)
                mlflow.log_metric("conversation_rate", conversation_rate)
                mlflow.pyfunc.log_model(
                    python_model=agent, registered_model_name=model, artifact_path="mlruns"
                )

                Profit_avg = (
                    agent.data.reward 
                    .mean()
                    .round(2)
                )
                mlflow.log_metric("Avg Gross Profit", Profit_avg)

                # Calculate cumulative count of actions
                df = agent.data.copy()
                actions = df['action'].unique()

                # Initialize a DataFrame to store cumulative counts
                cumulative_counts = pd.DataFrame(index=df.index, columns=actions).fillna(0)

                # Calculate cumulative count for each action
                for action in actions:
                    cumulative_counts[action] = (df['action'] == action).cumsum()

                # Convert counts to fractions
                cumulative_fractions = cumulative_counts.div(cumulative_counts.sum(axis=1), axis=0)

                # Plotting
                fig, ax = plt.subplots(figsize=(10, 6))
                for action in actions:
                    plt.plot(cumulative_fractions.index, cumulative_fractions[action], label=f'Action {action}')

                ax.set_xlabel('Time')
                ax.set_ylabel('Fraction of Action Chosen')
                ax.set_title('Fraction of Each Action Over Time')
                ax.legend()
                mlflow.log_figure(fig, "action_fraction.png")
            results[model] = agent.data

    
        df = results['Random']
        results['Fixed'] = train_and_eva_fixed_5( env, df )

    return results




# ---------------- train model for appendix ---------------- #

def train_model_appendix(
    exp_name, n_data, arms_range, n_arm, features_choice, models_choice, retrain_mode, scenario
):
    experiment = mlflow.set_experiment(exp_name)
    results = {}

    train_data, train_env = get_data(features_choice, arms_range, n_arm, n_data, scenario)

    for model in models_choice:
        if model != 'Fixed':
            with mlflow.start_run(experiment_id=experiment.experiment_id):
                mlflow.log_param("features", features_choice)
                mlflow.log_param("arms", arms_range)
                mlflow.log_param("n_arm", n_arm)
                mlflow.log_param("n_data", n_data)
                mlflow.log_param("model", model)
              
                train_env.current_index = 4000
                
                agent = train_conversion_reward(
                    train_data.copy(),
                    train_env,
                    retrain_mode,
                    arms_range,
                    n_arm,
                    MODELS[model],
                )

                conversation_rate = agent.data.converted.mean().round(2)
                mlflow.log_metric("conversation_rate", conversation_rate)
                mlflow.pyfunc.log_model(
                    python_model=agent, registered_model_name=model, artifact_path="mlruns"
                )

                profit = (
                   agent.data.reward
                    .mean()
                    .round(2)
                )
                mlflow.log_metric("Avg Gross Profit", profit)

            results[model] = agent.data
        elif model == "Fixed":
            results[model] = train_and_eva_fixed( train_env)

    return results















# ================== prepare data functions for ploting ================== #

def prepare_hotel_data(results):
    prepared_data = {}
    for model_name, df in results.items():
        df["action"] = df["action"].round(2)
        df["ttv"] = df["reward"] * df["base_price"] * (1 + df["action"])

        agg_data = df.groupby(["hotel_level", "action"])["ttv"].mean().reset_index()
        prepared_data[model_name] = agg_data
    return prepared_data

def prepare_customer_data(results):
    prepared_data = {}
    for model_name, df in results.items():
        df["action"] = df["action"].round(2)
        df["ttv"] = df["reward"] * df["base_price"] * (1 + df["action"])

        agg_data = df.groupby(["customer_type", "action"])["ttv"].mean().reset_index()
        prepared_data[model_name] = agg_data
    return prepared_data


def on_action_change():
    st.session_state.selected_action = selected_action

