import altair as alt
import pandas as pd
import streamlit as st

from app import point


"""
# Welcome to Streamlit!

Edit `/app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""


def ui():
    with st.echo(code_location='below'):
        total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
        num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

        data = []

        points_per_turn = total_points / num_turns

        for current_position in range(total_points):
            data.append(point.new_point(current_position, points_per_turn, total_points))

        st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
                        .mark_circle(color='#0068c9', opacity=0.5)
                        .encode(x='x:Q', y='y:Q'))
