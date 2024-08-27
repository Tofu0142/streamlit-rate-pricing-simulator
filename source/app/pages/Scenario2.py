import streamlit as st
import numpy as np
import pandas as pd
import os 
import webbrowser
import plotly.express as px
import mlflow

from train.train import (train_model)
from plots import (create_cumulative_rewards_animation)
from app.pages.ScenarioBase import ExperimentBase

class ScenarioB(ExperimentBase):
    def __init__(self, scenario, train_model_func):
        self.scenario = scenario
        self.train_model_func = train_model_func  # Assign the specific training function
        super().__init__(scenario, train_model_func)

    

def SB_main(n_data, arms_range, n_arm, models_choice, features_choice, retrain_mode):
    
    scenarioB = ScenarioB(2, train_model)
    scenarioB.train_and_display_results(
        n_data = n_data,
        arms_range = arms_range,
        n_arm = n_arm,
        models_choice = models_choice,
        features_choice = features_choice,
        retrain_mode = retrain_mode,
        
    )
    with st.container():
        scenarioB.display_results(st.session_state , create_cumulative_rewards_animation)
