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





# ================== prepare data functions for ploting ================== #

def prepare_hotel_data(results):
    prepared_data = {}
    for model_name, df in results.items():
        df["action"] = df["action"].round(2)
        #df["ttv"] = df["reward"] * df["base_price"] * (1 + df["action"])

        agg_data = df.groupby(["hotel_level", "action"])["reward"].mean().reset_index()
        prepared_data[model_name] = agg_data
    return prepared_data

def prepare_customer_data(results):
    prepared_data = {}
    for model_name, df in results.items():
        df["action"] = df["action"].round(2)
       
        agg_data = df.groupby(["customer_type", "action"])["reward"].mean().reset_index()
        prepared_data[model_name] = agg_data
    return prepared_data


def on_action_change():
    st.session_state.selected_action = selected_action

