import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
import mlflow
from train.train import ( train_model1)
from plots import (create_cumulative_rewards_animation, create_cumulative_rewards_animation_5)
from app.pages.ScenarioBase import ExperimentBase

class ScenarioE(ExperimentBase):
    def __init__(self, scenario, train_model_func):
        self.scenario = scenario
        self.train_model_func = train_model_func  # Assign the specific training function
        super().__init__(scenario, train_model_func)

    def display_results_changable(self, session_state):
        st.session_state = session_state
        with st.container():
            st.subheader("Reward and Margin Distribution By Each Day and Model")
            tab1, tab2  = st.tabs(["By every data", "By every 2 days"])
            with tab1:
                if 'result_retrain_every_data' not in st.session_state:
                    st.write("Please train the model first")
                    st.empty()
                else:
                    if st.session_state.result_retrain_every_data:
                        # Create the plot
                        fig = self.plot_fig(st.session_state.result_retrain_every_data)
                        st.pyplot(fig, use_container_width=True)
            with tab2:
                if 'result_retrain_every_2_days' not in st.session_state:
                    st.write("Please train the model first")
                    st.empty()
                else:
                    if st.session_state.result_retrain_every_2_days:
                        # Create the plot
                        fig = self.plot_fig(st.session_state.result_retrain_every_2_days)
                        st.pyplot(fig, use_container_width=True)
    

    def calculate_averages(self, df):
        chunk_size = 2000
        num_chunks = len(df) // chunk_size

        average_rewards = []
        average_margin = []

        for i in range(num_chunks+1):
            chunk = df.iloc[i * chunk_size:(i + 1) * chunk_size]
            avg_reward = chunk['converted'].mean()
            avg_margin = chunk['reward'].mean()

            average_rewards.append(avg_reward)
            average_margin.append(avg_margin)

        return average_rewards, average_margin


    def plot_fig(self, session_state_data):
        fig, ax = plt.subplots(2, 1, figsize=(12, 6))

        # Iterate over each model's data
        for model_name, df in session_state_data.items():
            average_rewards, average_margin = self.calculate_averages(df)
            x_range = range(1, len(average_rewards) + 1)

            # Plotting average rewards for each model
            ax[0].plot(x_range, average_rewards, marker='o', label=f'Avg Reward - {model_name}')
            
            # Plotting average margin for each model
            ax[1].plot(x_range, average_margin, marker='o', label=f'Avg Gross Profit - {model_name}')

        # Set properties for the first subplot
        ax[0].set_xlabel('Days')
        ax[0].set_ylabel('Average Reward')
        ax[0].set_title('Average Reward Per Request within a single day')
        ax[0].grid(True)
        ax[0].legend()

        # Set properties for the second subplot
        ax[1].set_xlabel('Days')
        ax[1].set_ylabel('Average Gross Profit')
        ax[1].set_title('Average Gross Profit Per Request within a single day')
        ax[1].grid(True)
        ax[1].legend()

        plt.tight_layout()
        return fig


def SE_main(arms_range, n_arm, models_choice, retrain_mode):
    st.info("In this simulation, as it's a long run, we will fix training data to 2000,\
             but when the data is over date such as apw<0, we will delete it and add new data to the training set."
             "\n The features we use are base_price, apw, occupancy, night, hotel_level,  promotion."
            "\n Here, customer will be elastic at promotion, which will be affected by apw."
            "\n The smaller the apw, the smaller the promotion."
            ,icon="📃")
    scenarioE = ScenarioE(3, train_model1)
    scenarioE.train_and_display_results(
        arms_range = arms_range,
        n_arm = n_arm,
        models_choice = models_choice,
        retrain_mode = retrain_mode,
        
    )
    with st.container():
        scenarioE.display_results(st.session_state, create_cumulative_rewards_animation_5)

    scenarioE.display_results_changable(st.session_state)
    