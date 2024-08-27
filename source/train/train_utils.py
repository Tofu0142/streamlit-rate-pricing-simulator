import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import mlflow

# print(os.getcwd())
from environments.simulator_env_1 import SimulatorEnv1, LogisticModelFeature
from environments.simulator_env_2 import SimulatorEnv2
from environments.simulator_env_3 import SimulatorEnv3
from mab import SimulatorBanditModel
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy



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

PROMOTION = LogisticModelFeature(
    name="promotion",
    model_coefficient=-1 / 10,
    space="negative_binomial",
    mean=4,
)

CUSTOMER_TYPE = LogisticModelFeature(
    name="customer_type",
    model_coefficient=0.1,
    space="random",
    low=0,
    high=1,
)

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
    "UCB1": LearningPolicy.UCB1(),
    "EG": LearningPolicy.EpsilonGreedy(epsilon=0.2),
    "TS": LearningPolicy.ThompsonSampling(),
}


def get_data(features_choice, arms_range, n_arm, n_data, scenario):
    filtered_dict = {key: FEATURES[key] for key in features_choice}
    env = SimulatorEnv1(
        discrete_action_space=True,
        n_actions=n_arm,
        intercept=10.0,
        features=filtered_dict,
        n_reps=(4000+n_data),
        first_arm=arms_range[0],
        last_arm=arms_range[1],
        scenario=scenario,
    )

    training_size =4000
    res = []
    env.reset()
   
    observation = env.data.iloc[0].to_dict()
    while len(res) < training_size:
        action = env.action_space.sample()[0]
        new_observation, reward, terminated, truncated, info = env.step(action)
        res.append([observation, action, reward])
        observation = new_observation
    
    # observations_df = df["observation"].apply(pd.Series)
    # df = pd.concat([observations_df, df.drop("observation", axis=1)], axis=1)

    return res, env

def train_conversion_reward(data, env, retrain_mode, arms_range, n_arm, learning_policy):
    training_batch_size = 2000
    model = SimulatorBanditModel(
        first_arm=arms_range[0],
        last_arm=arms_range[1],
        n_arms=n_arm,
        learning_policy=learning_policy,
    )
    terminated = False
    observation = data[0][0]
    ind = 0
    while not terminated:
        if ind < 4000:
            observation, action, converted = data[ind]
            # new_observation, reward, terminated, truncated, info = env.step(action)
            # print(observation, action, reward)
            reward = observation['base_price'] * (action) * converted
            model.collect(observation, action, reward, converted)
            #observation = new_observation
            ind += 1
            
        else:
            action = model.choose_action()
            new_observation, converted, terminated, truncated, info = env.step(action)
            if "action" in info:
                _action = info["action"]
            else:
                _action = action
            reward = observation['base_price'] * (_action) * converted
            model.collect(observation, _action, reward, converted)
            observation = new_observation

        if retrain_mode == "retrain_every_data":
            if env.current_index > 0:
                model.train('converted')

        elif retrain_mode == "retrain_every_2_days":
            if (env.current_index > 0) and (
                env.current_index % training_batch_size == 0
            ):
                
                model.train('converted')

    return model




# this function is used to train the model for scenario 1-4
# the reward is gross profit rather than conversion.
def train_profit_reward_1(data, env, retrain_mode, arms_range, n_arm, learning_policy):
    training_batch_size = 2000
    model = SimulatorBanditModel(
        first_arm=arms_range[0],
        last_arm=arms_range[1],
        n_arms=n_arm,
        learning_policy=learning_policy,
    )
    terminated = False
    observation = data[0][0]
    ind = 0
    while not terminated:
        if ind < 4000:
            observation, action, converted = data[ind]
            # new_observation, reward, terminated, truncated, info = env.step(action)
            # print(observation, action, reward)
            reward = observation['base_price'] * (action) * converted
            model.collect(observation, action, reward, converted)
            #observation = new_observation
            ind += 1
            
        else:
            action = model.choose_action()
            new_observation, converted, terminated, truncated, info = env.step(action)
            if "action" in info:
                _action = info["action"]
            else:
                _action = action
            reward = observation['base_price'] * (_action) * converted
            model.collect(observation, _action, reward , converted)
            observation = new_observation

        if retrain_mode == "retrain_every_data":
            if env.current_index > 0:
                model.train('profit')

        elif retrain_mode == "retrain_every_2_days":
            if (env.current_index > 0) and (
                env.current_index % training_batch_size == 0
            ):
                
                model.train('profit')

    return model


def get_data_5( arms_range, n_arm):
    features = {
    "base_price": BASE_PRICE,
    "apw": APW,
    "occupancy": OCCOUPANCY,
    "night": NIGHT,
    "hotel_level": STARS,
    "promotion": PROMOTION,
    }   
    env = SimulatorEnv2(
        discrete_action_space=True,
        n_actions=n_arm,
        intercept=10.0,
        features= features,
        n_reps= 2000,
        first_arm=arms_range[0],
        last_arm=arms_range[1],
    )

    training_size =2000
    res = []
    env.reset()
   
    observation = env.data.iloc[0].to_dict()
    while len(res) < training_size:
        action = env.action_space.sample()[0]
        new_observation, reward, terminated, truncated, info = env.step(action, 0)
        res.append([observation, action, reward])
        observation = new_observation
    
    # observations_df = df["observation"].apply(pd.Series)
    # df = pd.concat([observations_df, df.drop("observation", axis=1)], axis=1)

    return res, env



# --------------------------------------------------------------------
# this function is used to train the model for scenario 5 and 6
def train_profit_reward_5(data, env, end, retrain_mode, arms_range, n_arm, learning_policy):

    terminated = False
    action = data[0][1]
    
    env.reset()
    count, ind = 1, 0
    model = SimulatorBanditModel(
        first_arm=arms_range[0],
        last_arm=arms_range[1],
        n_arms=n_arm,
        learning_policy=learning_policy,
    )
    observation = env.data.iloc[0].to_dict()
    if retrain_mode == "retrain_every_data":
        while count <= end:
            while not terminated:
                if count <= 1 and ind<2000:
                    observation, action, converted = data[ind]
                    reward = observation['base_price'] * (action) * converted
                    model.collect(observation, action, reward, converted)
                    ind +=1
                else:
                
                    action = model.choose_action()
                    new_observation, converted, terminated, truncated, info = env.step(action, count)
                    if "action" in info:
                        _action = info["action"]
                    else:
                        _action = action
                        reward = observation['base_price'] * (_action) * converted
                        model.collect(observation, _action, reward, converted)
                    observation = new_observation

                model.train('profit')
            env.reset()
            terminated = False
            count +=1 
    elif retrain_mode == "retrain_every_2_days":
        while count <= end:
            while not terminated:
                if count <= 1 and ind<2000:
                    observation, action, converted = data[ind]
                    reward = observation['base_price'] * (action) * converted
                    model.collect(observation, action, reward, converted)
                    ind +=1
                else:
                
                    action = model.choose_action()
                    new_observation, converted, terminated, truncated, info = env.step(action, count)
                    if "action" in info:
                        _action = info["action"]
                    else:
                        _action = action
                        reward = observation['base_price'] * (_action) * converted
                        model.collect(observation, _action, reward, converted)
                    observation = new_observation
                    
                
            
            count += 1
            env.reset()
            terminated = False
            if (count > 0) and (count % 2 == 0):
                model.train('profit')
        
    return model, env
 #------------------------ get_data_6 for scenario 6 ------------------------#
def get_data_6( arms_range, n_arm):
    features = {
    "base_price": BASE_PRICE,
    "apw": APW,
    "occupancy": OCCOUPANCY,
    "night": NIGHT,
    "hotel_level": STARS,
    "promotion": PROMOTION,
    }   
    env = SimulatorEnv3(
        discrete_action_space=True,
        n_actions=n_arm,
        intercept=10.0,
        features= features,
        n_reps= 2000,
        first_arm=arms_range[0],
        last_arm=arms_range[1],
    )

    training_size =2000
    res = []
    env.reset()
   
    observation = env.data.iloc[0].to_dict()
    while len(res) < training_size:
        action = env.action_space.sample()[0]
        new_observation, reward, terminated, truncated, info = env.step(action, 0)
        res.append([observation, action, reward])
        observation = new_observation
    
    # observations_df = df["observation"].apply(pd.Series)
    # df = pd.concat([observations_df, df.drop("observation", axis=1)], axis=1)

    return res, env













#--------------------------------------------------------------------

def train_and_eva_fixed( env, data= None):
    if data is None:
        df = env.data.copy()
    else:
        df = data.drop(['reward', 'converted'], axis=1)

    # set fixed action
    if 'hotel_level' in df.columns:
        conditions = [
            df['hotel_level'] < 2,
            df['hotel_level'].isin([2, 3]),
            df['hotel_level'] == 4,
            df['hotel_level'] == 5
        ]
        choices = [0.1, 0.15, 0.2, 0.25]
        df['action'] = np.select(conditions, choices, default=0)
    else:
        quartiles = df['base_price'].quantile([0.25, 0.5, 0.75])
        conditions = [
            df['base_price'] <= quartiles[0.25],
            df['base_price'].between(quartiles[0.25], quartiles[0.5]),
            df['base_price'].between(quartiles[0.5], quartiles[0.75]),
            df['base_price'] > quartiles[0.75]
        ]
        choices = [0.1, 0.15, 0.2, 0.25]
        df['action'] = np.select(conditions, choices, default=0)

    rewards = []
    converteds = []
    for index, row in df.iterrows():
        # set the current state from the row, excluding the action :
        env._current_state = row.drop('action').to_dict()
        action = row['action']  
        try:
            converted = env.get_conversion(action)
            reward = row['base_price'] * (action) * converted
            converteds.append(converted)
            rewards.append(reward)
        except ValueError as e:
            print(f"Error at row {index}: {e}")
            rewards.append(None)
    
    df['reward'] = rewards
    df['converted'] = converteds
    return df

def train_and_eva_fixed_5( env, data= None):
    if data is None:
        df = env.data.copy()
    else:
        df = data.drop(['reward', 'converted'], axis=1)

    # set fixed action
    if 'hotel_level' in df.columns:
        conditions = [
            df['hotel_level'] < 2,
            df['hotel_level'].isin([2, 3]),
            df['hotel_level'] == 4,
            df['hotel_level'] == 5
        ]
        choices = [0.1, 0.15, 0.2, 0.25]
        df['action'] = np.select(conditions, choices, default=0)
    else:
        quartiles = df['base_price'].quantile([0.25, 0.5, 0.75])
        conditions = [
            df['base_price'] <= quartiles[0.25],
            df['base_price'].between(quartiles[0.25], quartiles[0.5]),
            df['base_price'].between(quartiles[0.5], quartiles[0.75]),
            df['base_price'] > quartiles[0.75]
        ]
        choices = [0.1, 0.15, 0.2, 0.25]
        df['action'] = np.select(conditions, choices, default=0)

    rewards = []
    converteds = []
    for index, row in df.iterrows():
        # set the current state from the row, excluding the action :
        env._current_state = row.drop('action').to_dict()
        action = row['action']  
        try:
            converted = env.get_conversion(action, int(index//2000))
            reward = row['base_price'] * (action) * converted
            converteds.append(converted)
            rewards.append(reward)
        except ValueError as e:
            print(f"Error at row {index}: {e}")
            rewards.append(None)
    
    df['reward'] = rewards
    df['converted'] = converteds
    return df

