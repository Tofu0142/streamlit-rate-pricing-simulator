import streamlit as st 
st.set_option('deprecation.showPyplotGlobalUse', False)
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
sys.path.append('../')
from source.utils import get_data, train_and_eva, train_and_eva1, plot_cumulative_rewards, create_cumulative_rewards_animation

MODELS = {
    "UCB1" : LearningPolicy.UCB1(alpha=1.2),
    "EG" : LearningPolicy.EpsilonGreedy(epsilon=0.1),
    "TS" : LearningPolicy.ThompsonSampling(),
    "LinTS": LearningPolicy.LinTS()
}

FEATURES = [    "base_price",
                "apw",
                "occupancy",
                "night",
                "hotel_level"]

st.set_page_config(
    page_title="Pricing Simulator",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded"
)


# train the model
def train_model (exp_name, n_data, arms_range, n_arm, features_choice, models_choice):

    experiment = mlflow.set_experiment(exp_name)    
    results = {}

    for model in models_choice:
        with mlflow.start_run(experiment_id=experiment.experiment_id):
            mlflow.log_param("features", features_choice)
            mlflow.log_param("arms",  arms_range)
            mlflow.log_param("n_arm", n_arm)
            mlflow.log_param("n_data", n_data)
            mlflow.log_param("model", model)
            print( MODELS[model])
            agent  = train_and_eva1(features_choice, arms_range, n_arm, n_data, MODELS[model])
            
            conversation_rate = agent.data.reward.mean().round(2)
            mlflow.log_metric("conversation_rate", conversation_rate)
            mlflow.pyfunc.log_model(python_model =agent, registered_model_name=model, artifact_path= 'mlruns')
            
            TTV_avg = (agent.data.reward * agent.data.base_price * (1 + agent.data.action)).mean().round(2)
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

#=================================
def prepare_data(results):
    prepared_data = {}
    for model_name, df in results.items():
        df['action'] = df['action'].round(2)
        df['ttv'] = df['reward'] * df['base_price'] * (1 + df['action'])

        
        agg_data = df.groupby(['hotel_level', 'action'])['ttv'].mean().reset_index()
        prepared_data[model_name] = agg_data
    return prepared_data

def plot_reward_distribution(prepared_data, selected_action):
    # Combine data from all models for the selected action
    combined_data_list = []
    for model_name, df in prepared_data.items():
        filtered_df = df[df['action'] == selected_action].copy()
        filtered_df['model'] = model_name
        combined_data_list.append(filtered_df)

    combined_data = pd.concat(combined_data_list)
    # Create the plot
    fig = px.bar(combined_data, x='hotel_level', y='ttv', color='model', barmode='group')
    fig.update_layout(title='Average TTV Distribution by Hotel Level and Model', xaxis_title='Hotel Level', yaxis_title='Avg TTV')
    return fig

def on_action_change():
    st.session_state.selected_action = selected_action

# =======================    
def main():

    st.title("Pricing Simulator 🤖")

    ## streamlit ui
    # sidebar for hyperparameters tuning
    st.sidebar.title("Hyperparameters")
    n_data = st.sidebar.slider('Select the number of data', 100, 10000, 1000)
    arms_range = st.sidebar.slider('Select the range of actions', 0.0, 1.0, (0.2, 0.5))
    #st.write('You selected:', arms_range[0], arms_range[1])
    n_arm = st.sidebar.slider('Select the number of arms', 1, 100, 10)
    #st.write('You selected:', n_arm)
    model_options = list(MODELS.keys())
    models_choice = st.sidebar.multiselect("Select the model", model_options)
    if not models_choice:
        st.stop()

    features_choice = st.sidebar.multiselect("Select the features", FEATURES)
    if not features_choice:
        st.stop()



    # Launch Mlflow from Streamlit
    st.sidebar.title("Mlflow Tracking 🔎")    
    if st.sidebar.button("Launch 🚀"):
        open_mlflow_ui()
        st.sidebar.success("MLflow Server is Live! http://localhost:5000")
        open_browser("http://localhost:5000")


    exp_type = st.radio("Select the experiment type", ["New Experiment", "Existing Experiment"], horizontal = True)
    if exp_type == "New Experiment":
        exp_name = st.text_input("Enter the name for New Experiment")
    else:
        try:
            if os.path.exists('./mlruns'):
                exps = [i.name for i in mlflow.search_experiments()]
                exp_name = st.selectbox("Select Experienment", exps)
            else:
                st.warning("🚨 No Previous Experiments Found! Set New Experiment ⬆️")
        except:
            st.warning("🚨 No Previous Experiments Found! Set New Experiment ⬆️")
 
    # Training the model starts from here
    if st.button("Train Model"):
        with st.spinner('Training the model...'):
            st.session_state.results = train_model(exp_name, n_data, arms_range, n_arm, features_choice, models_choice)
        st.success('Trained !')
    if 'results' not in st.session_state or not st.session_state.results:
        st.stop()

    with st.container():
        
        st.title('Average Conversation Rate')
        # Slice the dataframes based on the selected range and store in a new dictionary
        fig = create_cumulative_rewards_animation(st.session_state.results, 'reward')

        # Display the animation in Streamlit
        st.plotly_chart(fig)
    with st.container():
       
        st.title('Average TTV')
        # Slice the dataframes based on the selected range and store in a new dictionary
        fig = create_cumulative_rewards_animation(st.session_state.results, 'ttv')

        # Display the animation in Streamlit
        st.plotly_chart(fig)

    with st.form('form'):
        step = (arms_range[1] - arms_range[0]) / (n_arm - 1) if n_arm > 1 else 1
        
        selected_action = st.slider("Select Action", arms_range[0], arms_range[1], step=step)

        submitted = st.form_submit_button('Show')
        plot_container = st.empty()

    
    if submitted:
        
        # Process the form submission
        if 'prepared_results' not in st.session_state:
            st.session_state.prepared_results = prepare_data(st.session_state.results)

        st.session_state.selected_action = selected_action
        fig = plot_reward_distribution(st.session_state.prepared_results, st.session_state.selected_action)
        plot_container.plotly_chart(fig)


if __name__ == '__main__':
    main()