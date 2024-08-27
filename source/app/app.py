import streamlit as st

st.set_option("deprecation.showPyplotGlobalUse", False)
import pandas as pd
import numpy as np
import subprocess
import os
import webbrowser
import plotly.express as px

# -------------------
from sklearn.model_selection import train_test_split

from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy

import mlflow
import sys

from utils import (
    get_data,
    train_and_eva,
    train_and_eva1,
)
from plots import plot_reward_distribution,create_cumulative_rewards_animation


MODELS = {
    "UCB1": LearningPolicy.UCB1(alpha=1.2),
    "EG": LearningPolicy.EpsilonGreedy(epsilon=0.1),
    "TS": LearningPolicy.ThompsonSampling(),
}

FEATURES = ["base_price", "apw", "occupancy", "night", "hotel_level"]

st.set_page_config(
    page_title="Pricing Simulator",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
)


# train the model
def train_model(
    exp_name, n_data, arms_range, n_arm, features_choice, models_choice, retrain_mode
):
    experiment = mlflow.set_experiment(exp_name)
    results = {}

    train_data, train_env = get_data(features_choice, arms_range, n_arm, n_data)

    for model in models_choice:
        with mlflow.start_run(experiment_id=experiment.experiment_id):
            mlflow.log_param("features", features_choice)
            mlflow.log_param("arms", arms_range)
            mlflow.log_param("n_arm", n_arm)
            mlflow.log_param("n_data", n_data)
            mlflow.log_param("model", model)
            # print( MODELS[model])
            print("before", MODELS[model], train_env.current_index)
            train_env.current_index = len(train_data)
            print("after", MODELS[model], train_env.current_index)
            agent = train_and_eva(
                train_data.copy(),
                train_env,
                retrain_mode,
                arms_range,
                n_arm,
                MODELS[model],
            )

            conversation_rate = agent.data.reward.mean().round(2)
            mlflow.log_metric("conversation_rate", conversation_rate)
            mlflow.pyfunc.log_model(
                python_model=agent, registered_model_name=model, artifact_path="mlruns"
            )

            TTV_avg = (
                (agent.data.reward * agent.data.base_price * (1 + agent.data.action))
                .mean()
                .round(2)
            )
            mlflow.log_metric("TTV_avg", TTV_avg)

        results[model] = agent.data

    return results


# fuction for opening MLFlow UI directly from the app
def open_mlflow_ui():
    # start the mlflow tracking server as a subprocess
    cmd = "mlflow ui --port 5000"
    subprocess.Popen(cmd, shell=True)


def open_browser(url):
    webbrowser.open_new_tab(url)


# =================================
def prepare_data(results):
    prepared_data = {}
    for model_name, df in results.items():
        df["action"] = df["action"].round(2)
        df["ttv"] = df["reward"] * df["base_price"] * (1 + df["action"])

        agg_data = df.groupby(["hotel_level", "action"])["ttv"].mean().reset_index()
        prepared_data[model_name] = agg_data
    return prepared_data







# =======================
def main():
    st.title("Pricing Simulator 🤖")

    ## streamlit ui
    # sidebar for hyperparameters tuning
    st.sidebar.title("Hyperparameters")
    n_data = st.sidebar.slider("Select the number of data", 100, 10000, 1000)
    arms_range = st.sidebar.slider("Select the range of actions", 0.0, 1.0, (0.2, 0.5))
    # st.write('You selected:', arms_range[0], arms_range[1])
    n_arm = st.sidebar.slider("Select the number of arms", 1, 100, 10)
    # st.write('You selected:', n_arm)
    model_options = list(MODELS.keys())
    models_choice = st.sidebar.multiselect("Select the model", model_options)
    if not models_choice:
        st.stop()

    features_choice = st.sidebar.multiselect("Select the features", FEATURES)
    if not features_choice:
        st.stop()

    retrain_mode = st.sidebar.radio(
        "Select the retrain mode", ["Retrain all data", "Retrain every 2 days"], index=0
    )

    # Launch Mlflow from Streamlit
    st.sidebar.title("Mlflow Tracking 🔎" )
    if st.sidebar.button("Launch 🚀"):
        open_mlflow_ui()
        st.sidebar.success("MLflow Server is Live! http://localhost:5000")
        open_browser("http://localhost:5000")

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
        with st.spinner("Training the model..."):
            st.session_state.results = train_model(
                exp_name,
                n_data,
                arms_range,
                n_arm,
                features_choice,
                models_choice,
                retrain_mode,
            )
        st.success("Trained !")
    if "results" not in st.session_state or not st.session_state.results:
        st.stop()

    with st.container():
        st.title("Average Conversation Rate")
        # Slice the dataframes based on the selected range and store in a new dictionary
        fig = create_cumulative_rewards_animation(st.session_state.results, "reward")

        # Display the animation in Streamlit
        st.plotly_chart(fig)
    with st.container():
        st.title("Average TTV")
        # Slice the dataframes based on the selected range and store in a new dictionary
        fig = create_cumulative_rewards_animation(st.session_state.results, "ttv")

        # Display the animation in Streamlit
        st.plotly_chart(fig)

    with st.form("form"):
        # Process the form submission
        if "prepared_results" not in st.session_state:
            st.session_state.prepared_results = prepare_data(st.session_state.results)
            print(st.session_state.prepared_results)
        step = (arms_range[1] - arms_range[0]) / (n_arm - 1) if n_arm > 1 else 1
        min_action = min(df["action"].min() for df in st.session_state.results.values())
        max_action = max(df["action"].max() for df in st.session_state.results.values())
        selected_action = st.slider("Select Action", min_action, max_action)

        submitted = st.form_submit_button("Show")
        plot_container = st.empty()

    if submitted:
        st.session_state.selected_action = selected_action
        fig = plot_reward_distribution(
            st.session_state.prepared_results, st.session_state.selected_action
        )
        plot_container.plotly_chart(fig)


"""
if __name__ == '__main__':
    main()
"""
