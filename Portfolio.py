import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, t
import matplotlib.pyplot as plt
import yfinance as yf

import warnings
import sys
from datetime import datetime, timedelta


warnings.filterwarnings("ignore")

class Portfolio:
    def __init__(self, tickers, weights, start_date, end_date, confidence_level=0.95, n_days=1, calculate_cvar=True, distribution="Normal"):
        self.tickers = tickers
        self.weights = np.array(weights)
        self.confidence_level = confidence_level
        self.start_date = start_date
        self.end_date = end_date
        self.n_days = n_days
        self.calculate_cvar = calculate_cvar
        self.distribution = distribution
        self.data, _ = self.download_data()
        self.portfolio_returns = self.calculate_portfolio_returns()
        self.portfolio_mean = self.portfolio_returns.mean()
        self.portfolio_std = self.calculate_port_std()

    def download_data(self):
        """Download historical price data for all portfolio tickers from Yahoo Finance."""
        data = pd.DataFrame()  # Initialize an empty DataFrame to store all valid data
        errors = []
        st.empty()

        for ticker in self.tickers:
            try:
                ticker_data = yf.download(ticker, start=self.start_date, end=self.end_date)['Close']
                ticker_data.dropna(inplace=True)

                if ticker_data.isnull().values.any():
                    st.markdown(f"<span style='color:red;'>Downloaded data for {ticker} contains missing values. Please check the ticker or date range</span>", unsafe_allow_html=True)
                    sys.exit()
                elif ticker_data.empty:
                    st.error(f"No data for {ticker}")
                    #st.markdown(f"<span style='color:red;'>No data for {ticker}</span>", unsafe_allow_html=True)
                    sys.exit()
                else:
                    data[ticker] = ticker_data  # Add the data for the valid ticker to the DataFrame
                
            except ValueError as e:
                errors.append(f"Invalid data or tickers provided for {ticker}: {str(e)}")
                
            except Exception as e:
                errors.append(f"An error occurred while downloading data for {ticker}: {str(e)}")
        
        # Display errors if any
        if errors:
            for error in errors:
                st.markdown(f"<span style='color:red;'>{error}</span>", unsafe_allow_html=True)
            sys.exit()
        # Return the consolidated DataFrame and any errors
        return data, errors

    def calculate_portfolio_returns(self):
        """Calculate daily returns for the portfolio based on asset weights and returns."""
        if self.data is None:
            self.error_message = "Data could not be downloaded. Please check the tickers and date range."
            return None
        
        # Calculate daily returns
        daily_returns = self.data.pct_change().dropna()
        
        if daily_returns.empty:
            self.error_message = "The data is insufficient for calculating daily returns. Please check the tickers and date range."
            return None
        
        if daily_returns.isnull().values.any():
            self.error_message = "The calculated daily returns contain missing values."
            return None
        
        # Calculate portfolio returns
        portfolio_returns = daily_returns.dot(self.weights)
        return portfolio_returns

    def calculate_port_std(self):
        covmat = self.data.pct_change().dropna().cov()
        return np.sqrt(np.dot(self.weights.T ,np.dot(covmat, self.weights)))

    def historical_var(self):
        """Calculate VaR using Historical Simulation (in percentage terms) for n days."""
        var_percent = np.percentile(self.portfolio_returns, (1 - self.confidence_level) * 100)
        return var_percent * 100 * np.sqrt(self.n_days)

    def historical_cvar(self):
        """Calculate CVaR using Historical Simulation (in percentage terms) for n days."""
        var_threshold = np.percentile(self.portfolio_returns, (1 - self.confidence_level) * 100)
        cvar_returns = self.portfolio_returns[self.portfolio_returns <= var_threshold]
        return cvar_returns.mean() * 100 * np.sqrt(self.n_days)


    def parametric_var(self):
        """Calculate VaR using Parametric method (in percentage terms) for n days."""
        mean = self.portfolio_returns.mean()
        print('mean', mean)
        std_dev = self.calculate_port_std()#self.portfolio_returns.std()
        print('std', std_dev)
        print('ci', 1-self.confidence_level)
        if self.distribution == 'Normal':
            var_percent = norm.ppf(self.confidence_level) * std_dev * np.sqrt(self.n_days) - mean * self.n_days
        elif self.distribution == 'Student-t':
            dof, _, _ = t.fit(self.portfolio_returns)
            dof = np.round(dof)
            print(t.ppf(self.confidence_level, dof), "HERE")
            var_percent = np.sqrt((dof-2)/dof) * t.ppf(self.confidence_level, dof) * std_dev * np.sqrt(self.n_days) - mean * self.n_days
        else:
            raise ValueError('Pick either "normal" or "t"')
        return var_percent * 100

    def parametric_cvar(self):
        """Calculate CVaR (Conditional Value at Risk) using the parametric method for n days."""
        mean = self.portfolio_returns.mean()
        std_dev = self.calculate_port_std()#self.portfolio_returns.std()
        
        if self.distribution == 'Normal':
            z_alpha = norm.ppf(1 - self.confidence_level)
            cvar_percent = (norm.pdf(z_alpha) / (1 - self.confidence_level)) * std_dev * np.sqrt(self.n_days) - mean * self.n_days

        elif self.distribution == 'Student-t':
            dof, _, _ = t.fit(self.portfolio_returns)
            dof = np.round(dof)
            xanu = t.ppf(1 - self.confidence_level, dof)
            cvar_percent = (-1 / (1 - self.confidence_level)) * (1 / (1 - dof)) * (dof - 2 + xanu**2) * t.pdf(xanu, dof) * std_dev * np.sqrt(self.n_days) - mean * self.n_days
        else:
            raise ValueError('Pick either "normal" or "t" for the distribution parameter.')
        
        return cvar_percent * 100

    def monte_carlo_cvar(self, simulations=10000):
        """Calculate CVaR using Monte Carlo Simulation (in percentage terms) for n days."""
        mean = self.portfolio_returns.mean()
        std_dev = self.calculate_port_std()#self.portfolio_returns.std()
        
        # Simulate portfolio returns
        simulated_returns = np.random.normal(mean * self.n_days, std_dev * np.sqrt(self.n_days), simulations)
        
        # Calculate VaR at the given confidence level
        var_percent = np.percentile(simulated_returns, (1 - self.confidence_level) * 100)
        
        # Calculate CVaR, which is the average of the worst losses (those worse than VaR)
        cvar_percent = simulated_returns[simulated_returns <= var_percent].mean()
        
        return cvar_percent * 100
    
    def monte_carlo_var(self, simulations=10000):
        """Calculate VaR using Monte Carlo Simulation (in percentage terms) for n days."""
        mean = self.portfolio_returns.mean()
        std_dev = self.calculate_port_std()# self.portfolio_returns.std()
        simulated_returns = np.random.normal(mean * self.n_days, std_dev * np.sqrt(self.n_days), simulations)
        var_percent = np.percentile(simulated_returns, (1 - self.confidence_level) * 100)
        return var_percent * 100

    def plot_returns_distribution(self):
        """Plot the distribution of portfolio returns with VaR thresholds highlighted."""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(self.portfolio_returns, bins=50, alpha=0.5, color="blue", edgecolor="black", density=True, label="Portfolio Returns")

        # Calculate and plot VaR threshold lines
        historical_var = self.historical_var()
        parametric_var_normal = self.parametric_var()
        parametric_var_t = self.parametric_var()
        monte_carlo_var = self.monte_carlo_var()

        ax.axvline(x=historical_var / 100, color="red", linestyle="--", label=f"Historical VaR ({self.confidence_level*100:.0f}%)")
        ax.axvline(x=parametric_var_normal / 100, color="green", linestyle="--", label="Parametric VaR (Normal)")
        ax.axvline(x=parametric_var_t / 100, color="purple", linestyle="--", label="Parametric VaR (t-distribution)")
        ax.axvline(x=monte_carlo_var / 100, color="orange", linestyle="--", label="Monte Carlo VaR")

        ax.set_title(f"Portfolio Returns Distribution with VaR Confidence Intervals over {self.n_days} days")
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Frequency")
        ax.legend()

        # Display the plot using Streamlit
        st.pyplot(fig)

