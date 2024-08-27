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


from train.train_utils import *
import app.settings  as settings

MODELS = {
    "Random": LearningPolicy.Random(),
    "UCB1": LearningPolicy.UCB1(alpha=1.2),
    "EG": LearningPolicy.EpsilonGreedy(epsilon=0.1),
    "TS": LearningPolicy.ThompsonSampling(),
}

FEATURES = ["base_price", "apw", "occupancy", "night", "hotel_level", "customer_type", "promotion"]

st.set_page_config(
    page_title="Pricing Simulator",
    page_icon="🧊",
    layout = 'wide'
    
)

# fuction for opening MLFlow UI directly from the app
def open_mlflow_ui():
    # start the mlflow tracking server as a subprocess
    cmd = "mlflow ui --port 5000"
    subprocess.Popen(cmd, shell=True)


def open_browser(url):
    webbrowser.open_new_tab(url)


from .pages import Scenario1, Scenario2, Scenario3, Scenario4, Scenario5, Scenario6,Appendix

def main():
    import streamlit as st

    st.title("Pricing Simulator 🤖")

    st.markdown(
    """
    Currently, there are 5 scenarios defined:

    - 'Scenario 1': All customers are price inelastic

    - 'Scenario 2': All customers are price elastic

    - 'Scenario 3': customers who are looking at 5-star hotels are price inelastic, customers who are 
    < 5star are price elastic

    - 'Scenario 4': international customers are price inelastic, and domestic are price elastic

    - 'Scenario 5': all customers are price elastic at X value, and after every 3 days their elasticity value increases by Y amount, and we run this long enough
    """)

    
    choice = st.sidebar.radio("Go to", ["🏠 Home page","📖 Scenario 1", "✏️ Scenario 2", ":books: Scenario 3", "💡 Scenario 4", "⏳ Scenario 5", "🔮 Scenario 6", "🤡 Appendix"])

    st.sidebar.title("Hyperparameters")
    n_data = st.sidebar.slider("Select the number of data", 100, 30000, 1000)
    arms_range = st.sidebar.slider("Select the range of actions", 0.0, 1.0, (0.2, 0.5))
    # st.write('You selected:', arms_range[0], arms_range[1])
    n_arm = st.sidebar.slider("Select the number of arms", 1, 100, 10)
    # st.write('You selected:', n_arm)
    model_options = list(MODELS.keys()) + ["Fixed"]
    models_choice = st.sidebar.multiselect("Select the model", model_options)
    if not models_choice:
        st.stop()

    features_choice = st.sidebar.multiselect("Select the features", FEATURES)
    if not features_choice:
        st.stop()

    retrain_mode = st.sidebar.radio(
        "Select the retrain mode", ["Retrain every data", "Retrain every 2 days"], index=0
    )
    if retrain_mode == "Retrain every data":
        retrain_mode = "retrain_every_data"
    else:
        retrain_mode = "retrain_every_2_days"
    # print(retrain_mode)
    # Launch Mlflow from Streamlit
    st.sidebar.title("Mlflow Tracking 🔎" )
    if st.sidebar.button("Launch 🚀"):
        open_mlflow_ui()
        st.sidebar.success("MLflow Server is Live! {settings.mlflow_tracking_uri})")
        open_browser(settings.mlflow_tracking_uri)


    if choice == "📖 Scenario 1":
        Scenario1.SA_main(n_data, arms_range, n_arm, models_choice, features_choice, retrain_mode)
    
    if choice == "✏️ Scenario 2":
        Scenario2.SB_main(n_data, arms_range, n_arm, models_choice, features_choice, retrain_mode)
    if choice == ":books: Scenario 3":
        if 'hotel_level' in features_choice:
            Scenario3.SC_main(n_data, arms_range, n_arm, models_choice, features_choice, retrain_mode)
        else:
            st.warning("🚨 Please select 'hotel_level' as a feature")
    if choice == "💡 Scenario 4":
        if 'customer_type' in features_choice:
            Scenario4.SD_main(n_data, arms_range, n_arm, models_choice, features_choice, retrain_mode)
        else:
            st.warning("🚨 Please select 'customer_type' as a feature")
    if choice == "⏳ Scenario 5":
        Scenario5.SE_main(arms_range, n_arm, models_choice, retrain_mode)

    if choice == "🔮 Scenario 6":
        Scenario6.SF_main(arms_range, n_arm, models_choice, retrain_mode)

    if choice == "🤡 Appendix":
        Appendix.Appdendix_main(n_data, arms_range, n_arm, models_choice, features_choice, retrain_mode)

