import streamlit as st
import time
import numpy as np
import os
import mlflow
from train.train import (train_model)
from plots import (create_cumulative_rewards_animation)
from app.pages.ScenarioBase import ExperimentBase


class ScenarioA(ExperimentBase):
    def __init__(self, scenario, train_model_func):
        self.scenario = scenario
        self.train_model_func = train_model_func  # Assign the specific training function
        super().__init__(scenario, train_model_func)

    




#st.set_page_config(page_title="Scenario A", page_icon="📈")
def SA_main(n_data, arms_range, n_arm, models_choice, features_choice, retrain_mode):

    scenarioA = ScenarioA(1, train_model)
    scenarioA.train_and_display_results(
        n_data = n_data,
        arms_range = arms_range,
        n_arm = n_arm,
        models_choice = models_choice,
        features_choice = features_choice,
        retrain_mode = retrain_mode,
        
    )
    with st.container():
        scenarioA.display_results(st.session_state, create_cumulative_rewards_animation)


