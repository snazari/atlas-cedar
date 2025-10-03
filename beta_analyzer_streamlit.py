"""
Streamlit-compatible Beta Analysis Module
Adapted from compute_beta_v4.py for use in streamlit dashboard
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class StreamlitBetaAnalyzer:
    """Beta Analysis class optimized for Streamlit display."""
    
    def __init__(self, asset_name='Asset', confidence_level=0.95, risk_free_rate=0.0):
        self.asset_name = asset_name
        self.confidence_level = confidence_level
        self.risk_free_rate = risk_free_rate
        self.data = None
        self.results = {}
        
    def load_data_from_db(self, df):
        """
        Load data from portfolio database format.
        Expected columns: timestamp, current_value (price), initial_value (portfolio)
        """
        # Rename columns to match expected format
        df = df.copy()
        df = df.rename(columns={
            'timestamp': 'date',
            'initial_value': 'close',  # Asset price
            'current_value': 'results'  # Portfolio value
        })
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate returns
        df[f'{self.asset_name}_returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['results'].pct_change()
        
        # Remove NaN values
        df = df.dropna()
        
        self.data = df
        return df
    
    def calculate_beta_alpha(self):
        """Calculate beta and alpha metrics."""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        strategy_ret = self.data['strategy_returns'].dropna()
        market_ret = self.data[f'{self.asset_name}_returns'].dropna()
        
        # Align the series
        aligned_data = pd.DataFrame({
            'strategy': strategy_ret,
            'market': market_ret
        }).dropna()
        
        strategy_ret = aligned_data['strategy']
        market_ret = aligned_data['market']
        
        if len(aligned_data) < 30:
            raise ValueError("Insufficient data points for reliable calculation")
        
        # OLS regression
        X = market_ret.values.reshape(-1, 1)
        y = strategy_ret.values
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        try:
            coefficients = np.linalg.solve(X_with_intercept.T @ X_with_intercept,
                                         X_with_intercept.T @ y)
            alpha_ols = coefficients[0]
            beta_ols = coefficients[1]
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
            alpha_ols = coefficients[0]
            beta_ols = coefficients[1]
        
        # Calculate correlation and R-squared
        correlation = np.corrcoef(strategy_ret, market_ret)[0, 1]
        r_squared = correlation ** 2
        
        # Standard errors
        n = len(aligned_data)
        residuals = strategy_ret - (alpha_ols + beta_ols * market_ret)
        mse = np.sum(residuals**2) / (n - 2)
        
        market_ret_centered = market_ret - np.mean(market_ret)
        se_beta = np.sqrt(mse / np.sum(market_ret_centered**2))
        
        t_critical = stats.t.ppf((1 + self.confidence_level) / 2, n - 2)
        beta_ci_lower = beta_ols - t_critical * se_beta
        beta_ci_upper = beta_ols + t_critical * se_beta
        
        # Alpha calculations
        strategy_mean_return = np.mean(strategy_ret)
        market_mean_return = np.mean(market_ret)
        regular_alpha = strategy_mean_return - market_mean_return
        jensens_alpha = strategy_mean_return - (self.risk_free_rate + beta_ols * (market_mean_return - self.risk_free_rate))
        
        # Infer frequency for annualization
        time_diff = (self.data['date'].iloc[-1] - self.data['date'].iloc[0]).total_seconds()
        avg_period_seconds = time_diff / len(self.data)
        
        if avg_period_seconds < 600:  # < 10 minutes
            periods_per_year = 365 * 24 * 12  # 5-min data
        elif avg_period_seconds < 3600:  # < 1 hour
            periods_per_year = 365 * 24  # hourly
        else:
            periods_per_year = 365  # daily
        
        # Annualized values
        regular_alpha_annual = regular_alpha * periods_per_year
        jensens_alpha_annual = jensens_alpha * periods_per_year
        alpha_ols_annual = alpha_ols * periods_per_year
        
        self.results['beta_analysis'] = {
            'beta': beta_ols,
            'alpha_ols': alpha_ols,
            'alpha_ols_annual': alpha_ols_annual,
            'regular_alpha': regular_alpha,
            'regular_alpha_annual': regular_alpha_annual,
            'jensens_alpha': jensens_alpha,
            'jensens_alpha_annual': jensens_alpha_annual,
            'correlation': correlation,
            'r_squared': r_squared,
            'beta_ci_lower': beta_ci_lower,
            'beta_ci_upper': beta_ci_upper,
            'n_observations': n,
            'periods_per_year': periods_per_year
        }
        
        return self.results['beta_analysis']
    
    def calculate_performance_ratios(self):
        """Calculate performance ratios."""
        strategy_ret = self.data['strategy_returns'].dropna()
        market_ret = self.data[f'{self.asset_name}_returns'].dropna()
        beta = self.results['beta_analysis']['beta']
        periods_per_year = self.results['beta_analysis']['periods_per_year']
        annualization_factor = np.sqrt(periods_per_year)
        
        # Sharpe Ratio
        sharpe_ratio = np.mean(strategy_ret) / np.std(strategy_ret, ddof=1) if np.std(strategy_ret) != 0 else 0
        sharpe_ratio_annual = sharpe_ratio * annualization_factor
        
        # Sortino Ratio
        downside_returns = strategy_ret[strategy_ret < self.risk_free_rate]
        downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0
        sortino_ratio = np.mean(strategy_ret) / downside_std if downside_std != 0 else np.inf
        sortino_ratio_annual = sortino_ratio * annualization_factor
        
        # Treynor Ratio
        treynor_ratio = np.mean(strategy_ret) / beta if beta != 0 else np.inf
        treynor_ratio_annual = treynor_ratio * periods_per_year
        
        # Information Ratio
        excess_returns = strategy_ret - market_ret
        tracking_error = np.std(excess_returns, ddof=1)
        information_ratio = np.mean(excess_returns) / tracking_error if tracking_error != 0 else 0
        information_ratio_annual = information_ratio * annualization_factor
        
        # Drawdown calculations
        strategy_values = self.data['results']
        strategy_cumulative = strategy_values / strategy_values.iloc[0]
        strategy_running_max = strategy_cumulative.expanding().max()
        strategy_drawdown = (strategy_cumulative - strategy_running_max) / strategy_running_max
        max_drawdown_strategy = strategy_drawdown.min()
        
        asset_prices = self.data['close']
        asset_cumulative = asset_prices / asset_prices.iloc[0]
        asset_running_max = asset_cumulative.expanding().max()
        asset_drawdown = (asset_cumulative - asset_running_max) / asset_running_max
        max_drawdown_asset = asset_drawdown.min()
        
        # Initial drawdown
        initial_value_strategy = strategy_values.iloc[0]
        initial_value_asset = asset_prices.iloc[0]
        dd_init_strategy = np.max(initial_value_strategy - strategy_values) / initial_value_strategy
        dd_init_asset = np.max(initial_value_asset - asset_prices) / initial_value_asset
        
        # Total gains
        strategy_gain = (strategy_values.iloc[-1] - initial_value_strategy) / initial_value_strategy
        market_gain = (asset_prices.iloc[-1] - initial_value_asset) / initial_value_asset
        
        # Calmar Ratio
        annual_return = np.mean(strategy_ret) * periods_per_year
        calmar_ratio = annual_return / abs(max_drawdown_strategy) if max_drawdown_strategy != 0 else np.inf
        
        self.results['performance_ratios'] = {
            'sharpe_ratio_annual': sharpe_ratio_annual,
            'sortino_ratio_annual': sortino_ratio_annual,
            'treynor_ratio_annual': treynor_ratio_annual,
            'information_ratio_annual': information_ratio_annual,
            'calmar_ratio': calmar_ratio,
            'max_drawdown_strategy': max_drawdown_strategy,
            'max_drawdown_asset': max_drawdown_asset,
            'initial_drawdown_strategy': dd_init_strategy,
            'initial_drawdown_asset': dd_init_asset,
            'strategy_gain': strategy_gain,
            'market_gain': market_gain,
            'tracking_error_annual': tracking_error * annualization_factor
        }
        
        return self.results['performance_ratios']
    
    def create_scatter_plot(self):
        """Create scatter plot of strategy vs market returns."""
        strategy_ret = self.data['strategy_returns'].dropna()
        market_ret = self.data[f'{self.asset_name}_returns'].dropna()
        
        beta = self.results['beta_analysis']['beta']
        alpha = self.results['beta_analysis']['alpha_ols']
        
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=market_ret,
            y=strategy_ret,
            mode='markers',
            name='Returns',
            marker=dict(size=5, color='blue', opacity=0.5)
        ))
        
        # Regression line
        x_line = np.linspace(market_ret.min(), market_ret.max(), 100)
        y_line = alpha + beta * x_line
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name=f'Regression Line (β={beta:.4f})',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f'{self.asset_name} - Strategy vs Market Returns',
            xaxis_title='Market Returns',
            yaxis_title='Strategy Returns',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_cumulative_returns_plot(self):
        """Create cumulative returns comparison plot."""
        strategy_cum = (1 + self.data['strategy_returns']).cumprod()
        market_cum = (1 + self.data[f'{self.asset_name}_returns']).cumprod()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.data['date'],
            y=strategy_cum,
            mode='lines',
            name='Strategy',
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.data['date'],
            y=market_cum,
            mode='lines',
            name=f'{self.asset_name} (Market)',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Cumulative Returns Comparison',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_drawdown_plot(self):
        """Create drawdown analysis plot."""
        strategy_values = self.data['results']
        strategy_cumulative = strategy_values / strategy_values.iloc[0]
        strategy_running_max = strategy_cumulative.expanding().max()
        strategy_drawdown = (strategy_cumulative - strategy_running_max) / strategy_running_max * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.data['date'],
            y=strategy_drawdown,
            mode='lines',
            fill='tozeroy',
            line=dict(color='red'),
            name='Drawdown %'
        ))
        
        fig.update_layout(
            title='Drawdown Analysis',
            xaxis_title='Date',
            yaxis_title='Drawdown %',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_return_distribution_plot(self):
        """Create return distribution comparison."""
        strategy_ret = self.data['strategy_returns'].dropna()
        market_ret = self.data[f'{self.asset_name}_returns'].dropna()
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=strategy_ret,
            name='Strategy Returns',
            opacity=0.7,
            nbinsx=50,
            marker_color='blue'
        ))
        
        fig.add_trace(go.Histogram(
            x=market_ret,
            name=f'{self.asset_name} Returns',
            opacity=0.7,
            nbinsx=50,
            marker_color='orange'
        ))
        
        fig.update_layout(
            title='Return Distribution Comparison',
            xaxis_title='Returns',
            yaxis_title='Frequency',
            template='plotly_white',
            barmode='overlay',
            height=400
        )
        
        return fig
    
    def create_risk_return_plot(self):
        """Create risk-return profile plot."""
        strategy_ret = self.data['strategy_returns'].dropna()
        market_ret = self.data[f'{self.asset_name}_returns'].dropna()
        
        periods_per_year = self.results['beta_analysis']['periods_per_year']
        annualization_factor = np.sqrt(periods_per_year)
        
        annual_return_strategy = np.mean(strategy_ret) * periods_per_year
        annual_vol_strategy = np.std(strategy_ret, ddof=1) * annualization_factor
        annual_return_market = np.mean(market_ret) * periods_per_year
        annual_vol_market = np.std(market_ret, ddof=1) * annualization_factor
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[annual_vol_strategy, annual_vol_market],
            y=[annual_return_strategy, annual_return_market],
            mode='markers+text',
            marker=dict(size=15, color=['green', 'blue']),
            text=['Strategy', self.asset_name],
            textposition='top center',
            showlegend=False
        ))
        
        fig.update_layout(
            title='Risk-Return Profile (Annualized)',
            xaxis_title='Annual Volatility',
            yaxis_title='Annual Return',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_comprehensive_dashboard(self):
        """Create comprehensive 8-panel dashboard."""
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                'Beta Regression Analysis',
                'Cumulative Returns',
                'Return Distribution',
                'Risk-Return Profile',
                'Drawdown Analysis',
                'Price Chart',
                'Strategy Performance',
                'Market Comparison'
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )
        
        strategy_ret = self.data['strategy_returns'].dropna()
        market_ret = self.data[f'{self.asset_name}_returns'].dropna()
        aligned_data = pd.DataFrame({'strategy': strategy_ret, 'market': market_ret}).dropna()
        
        beta = self.results['beta_analysis']['beta']
        alpha = self.results['beta_analysis']['alpha_ols']
        
        # 1. Beta Regression
        x_range = np.linspace(aligned_data['market'].min(), aligned_data['market'].max(), 100)
        y_line = alpha + beta * x_range
        
        fig.add_trace(
            go.Scatter(
                x=aligned_data['market'],
                y=aligned_data['strategy'],
                mode='markers',
                marker=dict(size=4, opacity=0.6, color='blue'),
                name='Returns',
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_line,
                mode='lines',
                line=dict(color='red', width=2),
                name=f'β={beta:.4f}',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Cumulative Returns
        strategy_cum = (1 + strategy_ret).cumprod()
        market_cum = (1 + market_ret).cumprod()
        
        fig.add_trace(
            go.Scatter(
                x=self.data['date'],
                y=strategy_cum,
                mode='lines',
                line=dict(color='green', width=2),
                name='Strategy',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.data['date'],
                y=market_cum,
                mode='lines',
                line=dict(color='blue', width=2),
                name='Market',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Return Distribution
        fig.add_trace(
            go.Histogram(
                x=strategy_ret,
                opacity=0.7,
                nbinsx=50,
                marker_color='blue',
                name='Strategy',
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=market_ret,
                opacity=0.7,
                nbinsx=50,
                marker_color='orange',
                name='Market',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Risk-Return Profile
        periods_per_year = self.results['beta_analysis']['periods_per_year']
        annualization_factor = np.sqrt(periods_per_year)
        
        annual_return_strategy = np.mean(strategy_ret) * periods_per_year
        annual_vol_strategy = np.std(strategy_ret, ddof=1) * annualization_factor
        annual_return_market = np.mean(market_ret) * periods_per_year
        annual_vol_market = np.std(market_ret, ddof=1) * annualization_factor
        
        fig.add_trace(
            go.Scatter(
                x=[annual_vol_strategy, annual_vol_market],
                y=[annual_return_strategy, annual_return_market],
                mode='markers+text',
                marker=dict(size=15),
                text=['Strategy', self.asset_name],
                textposition='top center',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 5. Drawdown Analysis
        strategy_values = self.data['results']
        strategy_cumulative = strategy_values / strategy_values.iloc[0]
        strategy_running_max = strategy_cumulative.expanding().max()
        strategy_drawdown = (strategy_cumulative - strategy_running_max) / strategy_running_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=self.data['date'],
                y=strategy_drawdown,
                mode='lines',
                fill='tozeroy',
                line=dict(color='red'),
                name='Drawdown',
                showlegend=False
            ),
            row=3, col=1
        )
        
        # 6. Price Chart
        fig.add_trace(
            go.Scatter(
                x=self.data['date'],
                y=self.data['close'],
                mode='lines',
                line=dict(color='black', width=2),
                name='Price',
                showlegend=False
            ),
            row=3, col=2
        )
        
        # 7. Strategy Performance
        fig.add_trace(
            go.Scatter(
                x=self.data['date'],
                y=self.data['results'],
                mode='lines',
                line=dict(color='green', width=2),
                name='Portfolio Value',
                showlegend=False
            ),
            row=4, col=1
        )
        
        # 8. Market Comparison (normalized)
        strategy_norm = self.data['results'] / self.data['results'].iloc[0] * 100
        market_norm = self.data['close'] / self.data['close'].iloc[0] * 100
        
        fig.add_trace(
            go.Scatter(
                x=self.data['date'],
                y=strategy_norm,
                mode='lines',
                line=dict(color='green', width=2),
                name='Strategy',
                showlegend=False
            ),
            row=4, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=self.data['date'],
                y=market_norm,
                mode='lines',
                line=dict(color='blue', width=2),
                name='Market',
                showlegend=False
            ),
            row=4, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Comprehensive Beta Analysis Dashboard - {self.asset_name}",
            showlegend=False,
            height=1400,
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Market Returns", row=1, col=1)
        fig.update_yaxes(title_text="Strategy Returns", row=1, col=1)
        
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=2)
        
        fig.update_xaxes(title_text="Returns", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_xaxes(title_text="Annual Volatility", row=2, col=2)
        fig.update_yaxes(title_text="Annual Return", row=2, col=2)
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=3, col=1)
        
        fig.update_xaxes(title_text="Date", row=3, col=2)
        fig.update_yaxes(title_text="Price ($)", row=3, col=2)
        
        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=4, col=1)
        
        fig.update_xaxes(title_text="Date", row=4, col=2)
        fig.update_yaxes(title_text="Normalized (Base=100)", row=4, col=2)
        
        return fig
