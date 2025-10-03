import yaml
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

def load_config(config_file):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def get_crypto_data(crypto_id, days=365, interval='15m'):
    """Fetch cryptocurrency OHLCV data from Binance using ccxt."""
    exchange = ccxt.binance({'rateLimit': 1200, 'enableRateLimit': True})

    # Map crypto_id to binance symbol
    symbol_map = {
        'bitcoin': 'BTC/USDT',
        'ethereum': 'ETH/USDT'
    }
    symbol = symbol_map.get(crypto_id.lower())
    if not symbol:
        raise ValueError(f"Symbol for {crypto_id} not found.")

    # Convert interval to ccxt timeframe
    timeframe = interval
    if interval == 'daily':
        timeframe = '1d'

    # Calculate the starting timestamp
    since = exchange.parse8601(str(datetime.utcnow() - timedelta(days=days)))
    
    all_ohlcv = []
    
    print(f"Fetching {timeframe} data for {symbol}...")
    while since < exchange.milliseconds():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + exchange.parse_timeframe(timeframe) * 1000
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            break

    if not all_ohlcv:
        raise Exception(f"No data fetched for {symbol}. Please check the symbol and interval.")

    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'close']].rename(columns={'close': 'price'})
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp')
    
    return df

def get_periods_per_year(interval):
    """Calculate the number of intervals in a year."""
    if 'm' in interval:
        minutes = int(interval.replace('m', ''))
        return (60 / minutes) * 24 * 365
    elif 'h' in interval:
        hours = int(interval.replace('h', ''))
        return (24 / hours) * 365
    elif interval == 'daily' or interval == '1d':
        return 365
    elif interval == 'weekly':
        return 52
    else:
        # Default for unknown intervals, though ccxt has specific ones.
        return 1

def calculate_alpha_beta(returns1, returns2):
    """Calculate alpha and beta using linear regression"""
    # Create design matrix (add intercept term)
    X = np.column_stack([np.ones(len(returns2)), returns2])
    
    # Calculate coefficients using normal equation
    try:
        coefficients = np.linalg.lstsq(X, returns1, rcond=None)[0]
        alpha = coefficients[0]
        beta = coefficients[1]
        return alpha, beta
    except Exception as e:
        print(f"Error in linear regression: {e}")
        return None, None

def calculate_returns(prices):
    """Calculate daily returns"""
    returns = prices.pct_change().dropna()
    return returns

def create_plots(df1, df2, returns1, returns2, alpha, beta, crypto1_id, crypto2_id, interval):
    """Create interactive plots using Plotly"""
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            f'{crypto1_id} Price', f'{crypto2_id} Price',
            f'{crypto1_id} Returns', f'{crypto2_id} Returns',
            'Returns Regression', 'Returns Scatter'
        ),
        vertical_spacing=0.1,
        row_heights=[0.3, 0.3, 0.4]
    )
    
    # Plot 1: Crypto1 price
    fig.add_trace(
        go.Scatter(x=df1['timestamp'], y=df1['price'], mode='lines', name=f'{crypto1_id} Price'),
        row=1, col=1
    )
    
    # Plot 2: Crypto2 price
    fig.add_trace(
        go.Scatter(x=df2['timestamp'], y=df2['price'], mode='lines', name=f'{crypto2_id} Price'),
        row=1, col=2
    )
    
    # Plot 3: Crypto1 returns
    fig.add_trace(
        go.Scatter(x=returns1.index, y=returns1.values, mode='lines', name=f'{crypto1_id} Returns'),
        row=2, col=1
    )
    
    # Plot 4: Crypto2 returns
    fig.add_trace(
        go.Scatter(x=returns2.index, y=returns2.values, mode='lines', name=f'{crypto2_id} Returns'),
        row=2, col=2
    )
    
    # Plot 5: Regression line
    if alpha is not None and beta is not None:
        # Create regression line
        min_return = returns2.min()
        max_return = returns2.max()
        x_range = np.linspace(min_return, max_return, 100)
        y_range = alpha + beta * x_range
        
        fig.add_trace(
            go.Scatter(x=x_range, y=y_range, mode='lines', name=f'Regression Line (α={alpha:.4f}, β={beta:.4f})'),
            row=3, col=1
        )
        
        # Add scatter points for returns
        fig.add_trace(
            go.Scatter(x=returns2.values, y=returns1.values, mode='markers', 
                      name='Return Data Points', opacity=0.6),
            row=3, col=1
        )
        
        # Add scatter plot for the scatter plot view
        fig.add_trace(
            go.Scatter(x=returns2.values, y=returns1.values, mode='markers', 
                      name='Return Data Points', opacity=0.6),
            row=3, col=2
        )
        
        # Add regression line to scatter plot view
        fig.add_trace(
            go.Scatter(x=x_range, y=y_range, mode='lines', name=f'Regression Line'),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=f'Cryptocurrency Analysis: {crypto1_id} vs {crypto2_id} ({interval} intervals)',
        height=900,
        showlegend=False
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=2)
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Returns", row=2, col=1)
    
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_yaxes(title_text="Returns", row=2, col=2)
    
    fig.update_xaxes(title_text="Returns", row=3, col=1)
    fig.update_yaxes(title_text="Returns", row=3, col=1)
    
    fig.update_xaxes(title_text="Returns", row=3, col=2)
    fig.update_yaxes(title_text="Returns", row=3, col=2)
    
    return fig

def main():
    # Load configuration
    config = load_config('alpha_beta_experiment.yaml')
    
    # Get cryptocurrency data with specified interval
    crypto1_id = config['crypto1']['id']
    crypto2_id = config['crypto2']['id']
    interval = config.get('interval', 'daily')
    days = config.get('days', 365)
    
    print(f"Fetching data for {crypto1_id} and {crypto2_id} with {interval} intervals...")
    
    try:
        df1 = get_crypto_data(crypto1_id, days, interval)
        df2 = get_crypto_data(crypto2_id, days, interval)
        
        # Sort by timestamp before merging
        df1 = df1.sort_values('timestamp')
        df2 = df2.sort_values('timestamp')

        # Use merge_asof for robust time-series merging
        merged_df = pd.merge_asof(df1, df2, on='timestamp', suffixes=('_1', '_2'), direction='nearest')
        
        # Calculate returns
        returns1 = calculate_returns(merged_df['price_1'])
        returns2 = calculate_returns(merged_df['price_2'])
        
        # Calculate alpha and beta
        alpha, beta = calculate_alpha_beta(returns1, returns2)

        # Calculate Annualized Alpha
        periods_per_year = get_periods_per_year(interval)
        annualized_alpha = ((1 + alpha) ** periods_per_year) - 1
        
        # Create result dataframe
        results_df = pd.DataFrame({
            'crypto1': [crypto1_id],
            'crypto2': [crypto2_id],
            'alpha': [alpha],
            'annualized_alpha': [annualized_alpha],
            'beta': [beta],
            'days': [days],
            'interval': [interval],
            'calculation_date': [datetime.now().strftime('%Y-%m-%d')]
        })
        
        # Save to CSV
        output_file = f"{crypto1_id}_vs_{crypto2_id}_alpha_beta_{interval}.csv"
        results_df.to_csv(output_file, index=False)
        
        print(f"Results saved to {output_file}")
        print(f"Alpha ({interval}): {alpha:.6f}")
        print(f"Annualized Alpha: {annualized_alpha:.6f}")
        print(f"Beta: {beta:.6f}")

        # Create plots
        fig = create_plots(df1, df2, returns1, returns2, alpha, beta, crypto1_id, crypto2_id, interval)
        
        # Save interactive plot as HTML
        output_file = f"{crypto1_id}_vs_{crypto2_id}_alpha_beta_analysis_{interval}.html"
        fig.write_html(output_file)        
        print(f"Interactive plot saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
