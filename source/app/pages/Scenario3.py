import streamlit as st
import time
import numpy as np
import os
import mlflow
from utils import ( prepare_hotel_data)
from train.train import (train_model)
from plots import (create_cumulative_rewards_animation, plot_hotel_reward_distribution)
from app.pages.ScenarioBase import ExperimentBase

class ScenarioC(ExperimentBase):
    def __init__(self, scenario, train_model_func):
        self.scenario = scenario
        self.train_model_func = train_model_func  # Assign the specific training function
        super().__init__(scenario, train_model_func)
    
    def display_results_changable(self, session_state):
        st.session_state = session_state
        with st.container():
            st.subheader("Distribution by Hotel Level and Model")
            tab1, tab2  = st.tabs(["By every data", "By every 2 days"])
            with tab1:
                if 'result_retrain_every_data' not in st.session_state:
                    st.write("Please train the model first")
                    st.empty()
                    st.session_state_result_retrain_every_data = None
                else:
                    
                    # Process the form submission
                    if "prepared_results" not in st.session_state:
                        st.session_state.prepared_results = prepare_hotel_data(st.session_state.result_retrain_every_data)
                    
                    min_action = min(df["action"].min() for df in st.session_state.result_retrain_every_data.values())
                    max_action = max(df["action"].max() for df in st.session_state.result_retrain_every_data.values())
                    selected_action = st.slider("Select Action", min_action, max_action, key='action_slider_1')
                    
                    fig = plot_hotel_reward_distribution(
                        st.session_state.prepared_results, selected_action
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # Process the form submission
                if 'result_retrain_every_2_days' not in st.session_state:
                    st.write("Please train the model first")
                    st.empty()
                    st.session_state_result_retrain_every_2_days = None
                else:
                    if "prepared_results" not in st.session_state:
                        st.session_state.prepared_results = prepare_hotel_data(st.session_state.result_retrain_every_2_days)
                    
                    min_action = min(df["action"].min() for df in st.session_state.result_retrain_every_2_days.values())
                    max_action = max(df["action"].max() for df in st.session_state.result_retrain_every_2_days.values())
                    selected_action_2_days = st.slider("Select Action", min_action, max_action, key='action_slider_2')

                    fig = plot_hotel_reward_distribution(
                        st.session_state.prepared_results, selected_action_2_days
                    )
                    st.plotly_chart(fig, use_container_width=True)
            


#st.set_page_config(page_title="Scenario A", page_icon="📈")
def SC_main(n_data, arms_range, n_arm, models_choice, features_choice, retrain_mode):
    
    scenarioC = ScenarioC(3, train_model)
    scenarioC.train_and_display_results(
        n_data = n_data,
        arms_range = arms_range,
        n_arm = n_arm,
        models_choice = models_choice,
        features_choice = features_choice,
        retrain_mode = retrain_mode,
        
    )
    with st.container():
        scenarioC.display_results(st.session_state, create_cumulative_rewards_animation)

    scenarioC.display_results_changable(st.session_state)
    
    
    