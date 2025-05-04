import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import base64
import time
import yfinance as yf

if 'page' not in st.session_state:
    st.session_state.page = "main"

no_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(no_sidebar_style, unsafe_allow_html=True)

st.title("Portfolio Risk-Return Profile")
st.write("Please enter tickers and corresponding weights of your portfolio and press continue")

st.markdown("""
    <style>
        .stDataFrame {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

data = {
    "Ticker": ['AAPL', 'MSFT', 'NVDA', 'META'],
    "Weight": [0.25, 0.25, 0.25, 0.25]
}
initial_df = pd.DataFrame(data)

df = st.data_editor(
    initial_df,
    column_config={
        "Weight": st.column_config.NumberColumn(
            help="How much do you like this command (1-5)?",
            min_value=0,
            max_value=1,
            step=0.01
        ),
    },
    hide_index=True,
    width=500,
    use_container_width=True,
    num_rows='dynamic',
    disabled=False
)

st.session_state.input_df = df

col1, col2 = st.columns(2)
st.session_state.start_date = col1.date_input("Start Date", value=datetime.today().date()-timedelta(days=365))
st.session_state.end_date = col2.date_input("End Date", value=datetime.today().date(), max_value=datetime.today().date())
st.session_state.confidence_level = st.slider("VaR Confidence Level", 0.90, 0.99, 0.95)
st.session_state.distribution = st.selectbox("Select Distribution for Parametric VaR", ["Normal", "Student-t"])

if 'data_attempted' not in st.session_state:
    st.session_state['data_attempted'] = False

if 'use_csv' not in st.session_state:
    st.session_state['use_csv'] = False

if st.button('Continue', type='primary'):
    st.session_state['data_attempted'] = True

if st.session_state['data_attempted']:
    with st.spinner('Downloading...'):
        time.sleep(3)
        data = yf.download(
            'AAPL',
            start=datetime.today().date() - timedelta(days=365),
            end=datetime.today().date()
        )
    if data.empty:
        st.error('No data returned. The app may have hit the Yahoo Finance rate limit. Do you want to use the backup dataset?')
        if st.button("Use Backup Dataset", type="primary"):
            st.session_state['use_csv'] = True
    else:
        st.switch_page('pages/PortAnalysis.py')

if st.session_state['use_csv']:
    st.empty()
    st.switch_page("pages/Backup.py")

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

st.markdown("---")
st.write("Created by: Kerim Başbuğ")

github_logo = encode_image_to_base64('Icons/github.png')
linkedin_logo = encode_image_to_base64('Icons/linkedin.png')

st.markdown(f'<a href="https://github.com/kerimbasbug" target="_blank" style="text-decoration: none; color: inherit;"><img src="{github_logo}" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`kerimbasbug`</a>', unsafe_allow_html=True)
st.markdown(f'<a href="https://www.linkedin.com/in/kerimbasbug/" target="_blank" style="text-decoration: none; color: inherit;"><img src="{linkedin_logo}" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`kerimbasbug`</a>', unsafe_allow_html=True)
