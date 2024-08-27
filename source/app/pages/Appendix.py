import streamlit as st
import time
import numpy as np
import os
import mlflow
from train.train import (train_model_appendix)
from plots import (create_cumulative_rewards_animation)


def Appdendix_main(n_data, arms_range, n_arm, models_choice, features_choice, retrain_mode):
    exp_type = st.radio(
    "Select the experiment type",
    ["New Experiment", "Existing Experiment"],
    horizontal=True,
    )
    if exp_type == "New Experiment":
        exp_name = st.text_input("Enter the name for New Experiment")
    else:
        try:
            if os.path.exists("./mlruns"):
                exps = [i.name for i in mlflow.search_experiments()]
                exp_name = st.selectbox("Select Experienment", exps)
            else:
                st.warning("🚨 No Previous Experiments Found! Set New Experiment ⬆️")
        except:
            st.warning("🚨 No Previous Experiments Found! Set New Experiment ⬆️")
    
        # Training the model starts from here
    if st.button("Train Model"):
        result_key = f"result_s1_{retrain_mode}"
        with st.spinner("Training the model..."):
            st.session_state[result_key] = train_model_appendix(
                exp_name,
                n_data,
                arms_range,
                n_arm,
                features_choice,
                models_choice,
                retrain_mode,
                scenario=1
            )
        st.success("Trained !")
    if result_key not in st.session_state or not st.session_state[result_key]:
        st.stop()
    
    left, right = st.columns(2)
    with left:
        st.header("Retrain by every data")
        if 'result_s1_retrain_every_data' not in st.session_state:
            st.empty()
        else:
            if st.session_state.result_s1_retrain_every_data:
                with st.container():
                    st.caption("Average Conversation Rate")
                    fig = create_cumulative_rewards_animation(st.session_state.result_s1_retrain_every_data, "converted")

                    # Display the animation in Streamlit
                    st.plotly_chart(fig)
         
                with st.container():
                    st.caption("Average Gross Profit")
                    # Slice the dataframes based on the selected range and store in a new dictionary
                    fig = create_cumulative_rewards_animation(st.session_state.result_s1_retrain_every_data, "margin")

                    # Display the animation in Streamlit
                    st.plotly_chart(fig)




    with right:
        st.header("Retrain by every 2 days")
        if 'result_s1_retrain_every_2_days' not in st.session_state:
            st.empty()
        else:
  
            if st.session_state.result_s1_retrain_every_2_days:
                with st.container():
                    st.caption("Average Conversation Rate")
                    fig = create_cumulative_rewards_animation(st.session_state.result_s1_retrain_every_2_days, "converted")

                    # Display the animation in Streamlit
                    st.plotly_chart(fig, use_container_width=True)

            
                with st.container():
                    st.caption("Average Gross Profit")
                    # Slice the dataframes based on the selected range and store in a new dictionary
                    fig = create_cumulative_rewards_animation(st.session_state.result_s1_retrain_every_2_days, "margin")

                    # Display the animation in Streamlit
                    st.plotly_chart(fig, use_container_width=True)

