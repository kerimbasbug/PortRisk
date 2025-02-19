import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import yfinance as yf
from Portfolio import Portfolio
import quantstats as qs
from datetime import datetime
import time

st.title("Risk-Return Profile")

def check_df(df):
    err_msg = None
    df = df.dropna()
    df['Ticker'] = df['Ticker'].str.upper()
    valid_list = list(pd.read_csv('Data/companies_info.csv')['ticker'])
    df["In List"] = df["Ticker"].isin(valid_list)

    if df['Ticker'].duplicated().any():
        st.info('Duplicated Tickers Dropped')
        df = df.drop_duplicates('Ticker')

    if (df['In List'] == False).any():
        err_msg = "⚠️ Invalid ticker(s): "
        for i, ticker in enumerate(list(df[df['In List']==False]['Ticker'])):
            if i<len(list(df[df['In List']==False]['Ticker']))-1:
                err_msg += ticker + ", "
            else:
                err_msg += ticker + " "
        err_msg += "dropped and weights are readjusted."
        df = df[df['In List']]

    if (df['Weight'] < 0).any() or (df['Weight'] > 1).any():
        if err_msg is None:
            err_msg = '⚠️ Weights are readjusted.'
        df = df[df['Weight'] > 0]
        df = df[df['Weight'] <= 1]

    if np.sum(df['Weight']) != 1:
        if err_msg is None:
            err_msg = '⚠️ Weights are readjusted.'
        df['Weight'] = df['Weight']/df['Weight'].sum()
    
    if err_msg is None:
        err_msg = ""
    return df[['Ticker', 'Weight']], err_msg

df, err_msg = check_df(st.session_state.input_df)

if df.empty:
    st.warning('Please enter valid tickers. Redirecting to main page...')
    time.sleep(5)
    st.switch_page('main.py')

def stream_data():
    for word in err_msg.split(" "):
        yield word + " "
        time.sleep(0.05)

st.write_stream(stream_data)

port = Portfolio(tickers=df['Ticker'],
                 weights=df['Weight'],
                 start_date=st.session_state.start_date,
                 end_date=st.session_state.end_date,
                 confidence_level=st.session_state.confidence_level,
                 n_days=1,
                 calculate_cvar=True,
                 distribution=st.session_state.distribution)

pie_fig = px.pie(df, names='Ticker', values='Weight', title=' ', hole=0.40, color_discrete_sequence=px.colors.sequential.Magma)
pie_fig.update_layout(
    legend=dict(y=0.5, yanchor='middle'),
    margin=dict(t=0, b=0, l=0, r=0),
    height=220)

perf_fig = go.Figure(
    go.Scatter(
        x=port.portfolio_returns.index, 
        y=(1+port.portfolio_returns).cumprod(), 
        mode="lines", 
        name="Close Price"
    )
)
perf_fig.update_layout(
    height=220,
    yaxis_title="Portfolio Value",
    yaxis=dict(showticklabels=True, showline=False, showgrid=False),
    margin=dict(l=10, r=10, t=30, b=10),  # Tighten the layout
    hovermode=False,
    dragmode=False
)

col1, col2, col3 = st.columns([1,3,2])
col1.markdown('**<span style="color:rgba(15,17,22);">.</span>**', unsafe_allow_html=True)
col1.metric("Return", f"{qs.stats.cagr(port.portfolio_returns,compounded=False)*100:.1f}%", "")
col1.metric("Volatility", f"{qs.stats.volatility(port.portfolio_returns, annualize=True, periods=252)*100:.1f}%", "")
col2.plotly_chart(perf_fig, config={"displayModeBar": False})
col3.plotly_chart(pie_fig, use_container_width=True, config={"displayModeBar": False})

# -------
rf_rate = yf.download('^TNX', end=datetime.today(), progress=False)['Close'].iloc[-1].values[0]/100
#rf_rate = yf.download('^TNX', end=datetime.today(), progress=False)['Adj Close'].iloc[-1].values[0]/100
sharpe = qs.stats.sharpe(port.portfolio_returns, rf=rf_rate, periods=252)
sortino = qs.stats.sortino(port.portfolio_returns, rf=rf_rate, periods=252)
mdd = qs.stats.max_drawdown(port.portfolio_returns)
calmar = qs.stats.calmar(port.portfolio_returns)
treynor = qs.stats.treynor_ratio(port.portfolio_returns, benchmark='SPY', rf=rf_rate)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Sharpe", f"{sharpe:.2f}")
col2.metric("Sortino", f"{sortino:.2f}")
col3.metric("Calmar", f"{calmar:.2f}")
col4.metric("Max Drawdown", f"{-mdd*100:.2f}%")
col5.metric("Treynor", f"{treynor:.2f}")

# -------

bin_size = 0.001
ret_fig = ff.create_distplot(
    [port.portfolio_returns.values],
    ['Portfolio Returns'],
    bin_size=[bin_size],
    show_rug=False,
    colors=['lightblue'],
    histnorm='',
    show_curve=False
)

historical_var_result = np.round(np.abs(port.historical_var()),2)
historical_cvar_result = np.round(np.abs(port.historical_cvar()),2)

parametric_var_result = np.round(np.abs(port.parametric_var()),2)
parametric_cvar_result = np.round(np.abs(port.parametric_cvar()),2)

monte_carlo_var_result = np.round(np.abs(port.monte_carlo_var()),2)
monte_carlo_cvar_result = np.round(np.abs(port.monte_carlo_cvar()),2)

ret_fig.add_trace(
    go.Scatter(
        x=[-monte_carlo_var_result/100, -monte_carlo_var_result/100],
        y=[0,500],
        mode='lines',
        line=dict(color="rgb(146,55,124)", width=2, dash="dash"),
        name='Monte Carlo VaR'
    )
)

ret_fig.add_trace(
    go.Scatter(
        x=[-parametric_var_result/100, -parametric_var_result/100],
        y=[0,500],
        mode='lines',
        line=dict(color="orange", width=2, dash="dash"),
        name='Parametric VaR'
    )
)
ret_fig.add_trace(
    go.Scatter(
        x=[-historical_var_result/100,-historical_var_result/100],
        y=[0,500],
        mode='lines',
        line=dict(color="rgb(235,67,57)", width=2, dash="dash"),
        name='Hist. VaR'
    )
)

bin_edges = np.arange(
    np.min(port.portfolio_returns),
    np.max(port.portfolio_returns) + bin_size,
    bin_size
)
hist_counts, _ = np.histogram(port.portfolio_returns, bins=bin_edges)
max_count = np.max(hist_counts)

ret_fig.update_layout(
    legend_orientation='h',
    legend=dict(
        yanchor='bottom',
        xanchor='center',
        x=0.5,
        y=-0.25
    ),
    margin=dict(t=0, b=0, l=0, r=0),
    height=320,
    xaxis_range=[-np.maximum(np.abs(np.min(port.portfolio_returns.values)), np.abs(np.max(port.portfolio_returns.values)))-bin_size*5,
                 np.maximum(np.abs(np.min(port.portfolio_returns.values)), np.abs(np.max(port.portfolio_returns.values)))+bin_size*5],
    yaxis_range=[0, max_count],  # Set the y-axis range dynamically
    yaxis=dict(showticklabels=False, showline=False, showgrid=False),
    hovermode=False,
    dragmode=False
)

st.plotly_chart(ret_fig, use_container_width=True, config={"displayModeBar": False})

# -------

col, col1, col2, col3, _ = st.columns([3,3,3,3,1])
col.metric("", "VaR", delta="")
col1.metric("Historic VaR", str(historical_var_result)+ "%", "")
if st.session_state.distribution == 'Normal':
    col2.metric("VaR (Normal)", str(parametric_var_result)+ "%", "")
else:
    col2.metric("VaR (Student-t)", str(parametric_var_result)+ "%", "")
col3.metric("Monte Carlo VaR", str(monte_carlo_var_result)+ "%", "")

col, col1, col2, col3, _ = st.columns([3,3,3,3,1])
col.metric("", "CVaR", delta="")
col1.metric("Historic CVaR", str(historical_cvar_result)+ "%", "")
if st.session_state.distribution == 'Normal':
    col2.metric("CVaR (Normal)", str(parametric_cvar_result)+ "%", "")
else:
    col2.metric("CVaR (Student-t)", str(parametric_cvar_result)+ "%", "")
col3.metric("Monte Carlo CVaR", str(monte_carlo_cvar_result)+ "%", "")


parameters_markdown = f"""
<div style="display: flex; justify-content: space-between; width: 100%; align-items: center;">
    <div style="flex: 1; text-align: center;">{"1-Day Value At Risk"}</div>
    <div style="flex: 0; text-align: center;">|</div>
    <div style="flex: 1; text-align: center;">{f"Confidence Interval: {st.session_state.confidence_level*100}%"}</div>
</div>
<br>
"""
st.markdown(parameters_markdown, unsafe_allow_html=True)

# -------

if st.button("Restart", type='primary'):
    st.session_state.page = "main"
    st.switch_page('main.py')