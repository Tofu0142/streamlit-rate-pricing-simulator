import streamlit as st
import numpy as np
import pandas as pd
import os 
import webbrowser
import plotly.express as px
import mlflow

from train.train import (train_model)
from plots import (create_cumulative_rewards_animation)



class ExperimentBase:
    def __init__(self, scenario, train_model_func):
        self.scenario = scenario
        self.train_model_func = train_model_func  # Assign the specific training function
        
    def select_experiment(self):
        exp_type = st.radio(
            "Select the experiment type",
            ["New Experiment", "Existing Experiment"],
            horizontal=True,
        )
        if exp_type == "New Experiment":
            return st.text_input("Enter the name for New Experiment")
        else:
            try:
                if os.path.exists("./mlruns"):
                    exps = [i.name for i in mlflow.search_experiments()]
                    return st.selectbox("Select Experiment", exps)
                else:
                    st.warning("🚨 No Previous Experiments Found! Set New Experiment ⬆️")
            except:
                st.warning("🚨 No Previous Experiments Found! Set New Experiment ⬆️")
            return None

    def train_and_display_results(self, **kwargs):

        exp_name = self.select_experiment()
        if exp_name and st.button("Train Model"):
            result_key = f"result_{kwargs.get('retrain_mode', '')}"
            if result_key not in st.session_state:
                st.session_state[result_key] = None
            with st.spinner("Training the model..."):
                # Call the assigned training function
                st.session_state[result_key] = self.train_model_func(
                    exp_name=exp_name, scenario=self.scenario, **kwargs
                )
            st.success("Trained !")

        

    def display_results(self, session_state, visulization_func):
        
        # This method display the results of the training
        st.session_state = session_state

        left, right = st.columns(2)
        with left:
            st.header("Retrain by every data")
            if 'result_retrain_every_data' not in st.session_state:
                st.empty()
            else:
                if st.session_state.result_retrain_every_data:
                    with st.container():
                        st.caption("Average Conversation Rate Per Request")
                        fig = visulization_func(st.session_state.result_retrain_every_data, "converted")

                        # Display the animation in Streamlit
                        st.plotly_chart(fig)
            
                    with st.container():
                        st.caption("Average Gross Profit Per Request")
                        # Slice the dataframes based on the selected range and store in a new dictionary
                        fig = visulization_func(st.session_state.result_retrain_every_data, "margin")

                        # Display the animation in Streamlit
                        st.plotly_chart(fig)




        with right:
            st.header("Retrain by every 2 days")
            if 'result_retrain_every_2_days' not in st.session_state:
                st.empty()
            else:
    
                if st.session_state.result_retrain_every_2_days:
                    with st.container():
                        st.caption("Average Conversation Rate Per Request")
                        fig = visulization_func(st.session_state.result_retrain_every_2_days, "converted")

                        # Display the animation in Streamlit
                        st.plotly_chart(fig, use_container_width=True)

                
                    with st.container():
                        st.caption("Average Gross Profit Per Request")
                        # Slice the dataframes based on the selected range and store in a new dictionary
                        fig = visulization_func(st.session_state.result_retrain_every_2_days, "margin")

                        # Display the animation in Streamlit
                        st.plotly_chart(fig, use_container_width=True)


    def display_results_changable(self, session_state):
        # You can over write this method to display the results of the training
        raise NotImplementedError("This method should be over written in the child class")
