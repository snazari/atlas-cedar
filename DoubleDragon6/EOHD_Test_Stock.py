"""
EODHDClient - A Python client for the EODHD Financial Data API.

Author: mfarsh, snaz and DeepSeek :)  
"""

import requests
import pandas as pd
import pytz
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Any
import os
import yaml

# Function to get API key from config file
def get_api_key_from_config(config_file='config.yaml'):
    """
    Get API key from configuration file
    
    Args:
        config_file (str): Path to the configuration file
        
    Returns:
        str: API key or None if not found
    """
    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            try:
                config = yaml.safe_load(file)
                if config and 'api' in config and 'eohd_api_key' in config['api']:
                    return config['api']['eohd_api_key']
            except Exception as e:
                print(f"Error loading API key from config: {e}")
    return None


class EODHDClient:
    """
    Client for interacting with the EODHD Financial Data API.
    
    This class provides methods to fetch various financial data including
    historical prices, intraday data, and technical indicators.
    """
    
    BASE_URL = "https://eodhistoricaldata.com/api"
    
    def __init__(self, api_key: str = None):
        """
        Initialize the EODHD API client.
        
        Args:
            api_key (str): Your EODHD API key. If None, will attempt to load from config file
        """
        # Use provided API key or try to get from config
        if api_key is None or api_key == "":
            config_api_key = get_api_key_from_config()
            if config_api_key:
                self.api_key = config_api_key
            else:
                raise ValueError("No API key provided and none found in config.yaml")
        else:
            self.api_key = api_key
        
    def date_to_unix(self, date_str: str, date_format: str = "%d-%b-%Y", 
                     input_tz: str = "US/Eastern") -> int:
        """
        Converts a date string to a UNIX timestamp (seconds since 1970-01-01 UTC).
        
        Args:
            date_str (str): Input date string (e.g., "01-Jan-2020")
            date_format (str): Format matching date_str (default: "%d-%b-%Y")
            input_tz (str): Timezone of input date (default: "US/Eastern")
        
        Returns:
            int: UNIX timestamp
        """
        naive_dt = datetime.strptime(date_str, date_format)
        tz = pytz.timezone(input_tz)
        localized_dt = tz.localize(naive_dt)
        utc_dt = localized_dt.astimezone(pytz.utc)
        return int(utc_dt.timestamp())
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict:
        """
        Make a request to the EODHD API.
        
        Args:
            endpoint (str): API endpoint
            params (Dict[str, Any]): Query parameters
            
        Returns:
            Dict: API response as JSON
            
        Raises:
            ValueError: If the API returns an error
        """
        params['api_token'] = self.api_key
        params['fmt'] = params.get('fmt', 'json')
        
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            raise ValueError(f"API request failed: {response.status_code} - {response.text}")
            
        return response.json()
    
    def get_intraday_data(self, symbol: str, interval: str = "15m", 
                         start_date: Optional[Union[str, datetime]] = None,
                         end_date: Optional[Union[str, datetime]] = None,
                         timezone: str = "UTC") -> pd.DataFrame:
        """
        Fetch intraday price data for a specific symbol.
        
        Args:
            symbol (str): Stock symbol (e.g., "AAPL" or "AAPL.US")
            interval (str): Time interval (e.g., "5m", "15m", "1h")
            start_date: Start date as string (YYYY-MM-DD) or datetime object
            end_date: End date as string (YYYY-MM-DD) or datetime object
            timezone (str): Timezone for timestamp data (default: "UTC")
            
        Returns:
            pd.DataFrame: DataFrame with intraday price data with timezone-aware index
        """
        params = {'interval': interval}
        
        # Process dates
        if start_date:
            if isinstance(start_date, datetime):
                from_timestamp = int(start_date.timestamp())
            else:
                from_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            params['from'] = from_timestamp
            
        if end_date:
            if isinstance(end_date, datetime):
                to_timestamp = int(end_date.timestamp())
            else:
                to_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
            params['to'] = to_timestamp
        
        endpoint = f"intraday/{symbol}"
        data = self._make_request(endpoint, params)
        
        df = pd.DataFrame(data)
        if not df.empty and 'timestamp' in df.columns:
            # Convert timestamps to datetime and localize to specified timezone
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            if timezone != "UTC":
                df['timestamp'] = df['timestamp'].dt.tz_convert(timezone)
            df.set_index('timestamp', inplace=True)
        
        df.set_index("datetime", inplace=True)
        df.drop(['gmtoffset'],inplace=True,axis=1)
        return df
    
    def get_historical_prices(self, symbol: str, start_date: str, end_date: str,
                             period: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch daily historical price data.
        
        Args:
            symbol (str): Stock symbol (e.g., "AAPL" or "AAPL.US")
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            period (Optional[str]): Alternative to date range, e.g., "d" (daily), "w" (weekly)
            
        Returns:
            pd.DataFrame: DataFrame with historical price data
        """
        params = {
            'from': start_date,
            'to': end_date
        }
        
        if period:
            params['period'] = period
            
        endpoint = f"eod/{symbol}"
        data = self._make_request(endpoint, params)
        
        df = pd.DataFrame(data)
        
        # Handle case when timestamp is in the data
        if not df.empty and 'timestamp' in df.columns:
            # Convert timestamps to datetime and set as index
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('datetime', inplace=True)
            if 'gmtoffset' in df.columns:
                df.drop(['gmtoffset'], inplace=True, axis=1)
        
        # For API responses that include 'date' column
        elif not df.empty and 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
            df.set_index('datetime', inplace=True)
        
        # If neither datetime nor date columns exist, use default index
        elif not df.empty:
            print(f"Warning: No datetime or date column found for {symbol}. Using default index.")
        
        return df
    
    def get_dividends_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch stock dividend history.
        
        Args:
            symbol (str): Stock symbol (e.g., "AAPL" or "AAPL.US")
            
        Returns:
            pd.DataFrame: DataFrame with dividend history data
        """
        params = {'type': 'dividends'}
        endpoint = f"div/{symbol}"
        data = self._make_request(endpoint, params)
        
        return pd.DataFrame(data)
    
    def get_splits_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch stock split history.
        
        Args:
            symbol (str): Stock symbol (e.g., "AAPL" or "AAPL.US")
            
        Returns:
            pd.DataFrame: DataFrame with split history data
        """
        params = {'type': 'splits'}
        endpoint = f"splits/{symbol}"
        data = self._make_request(endpoint, params)
        
        return pd.DataFrame(data)
    
    def get_calendar_splits(self, symbol: str) -> pd.DataFrame:
        """
        Fetch future splits for a symbol.
        
        Args:
            symbol (str): Stock symbol (e.g., "AAPL" or "AAPL.US")
            
        Returns:
            pd.DataFrame: DataFrame with calendar data
        """
        params = {'type': 'splits'}
        endpoint = f"calendar/splits?api_token={self.api_key}&fmt=json"
        data = self._make_request(endpoint, params)
        
        return pd.DataFrame(data)
    
    def get_technical_indicator(self, symbol: str, function: str, 
                              period: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch technical indicator data.
        
        Args:
            symbol (str): Stock symbol (e.g., "AAPL" or "AAPL.US")
            function (str): Technical indicator function (e.g., "splitadjusted", "rsi")
            period (Optional[int]): Period for the indicator calculation
            
        Returns:
            pd.DataFrame: DataFrame with technical indicator data
        """
        params = {'function': function}
        
        if period:
            params['period'] = period
            
        endpoint = f"technical/{symbol}"
        data = self._make_request(endpoint, params)
        
        df = pd.DataFrame(data)
        return df
    
    def save_data_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save DataFrame to CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str): Output filename
        """
        df.to_csv(filename)
        print(f"Data saved to '{filename}' (rows: {len(df)})")
        
    def plot_closing_price(self, symbol: str, start_date: str, end_date: str, 
                          title: Optional[str] = None, 
                          use_candlestick: bool = False) -> None:
        """
        Plot the time history of closing prices for a symbol using Plotly.
        
        Args:
            symbol (str): Stock symbol (e.g., "AAPL" or "AAPL.US")
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            title (Optional[str]): Custom chart title (defaults to "{symbol} Closing Price")
            use_candlestick (bool): If True, plot candlestick chart instead of line (default: False)
        """
        # Get the historical price data
        df = self.get_historical_prices(symbol, start_date, end_date)
        
        if df.empty:
            print(f"No data found for {symbol} between {start_date} and {end_date}")
            return
        
        # Ensure date column is datetime for plotting
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        
        # Create figure
        if use_candlestick:
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=symbol
            )])
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['close'],
                mode='lines',
                name=f"{symbol} Close"
            ))
        
        # Set title
        chart_title = title if title else f"{symbol} Closing Price"
        fig.update_layout(
            title=chart_title,
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_white'
        )
        
        # Display the plot
        fig.show()
        
    def plot_split_adjusted_price(self, symbol: str, start_date: str, end_date: str,
                                 title: Optional[str] = None, 
                                 include_volume: bool = False) -> None:
        """
        Plot the split-adjusted price history for a symbol using Plotly.
        
        Args:
            symbol (str): Stock symbol (e.g., "AAPL" or "AAPL.US")
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            title (Optional[str]): Custom chart title (defaults to "{symbol} Split-Adjusted Price")
            include_volume (bool): If True, include volume subplot (default: False)
        """
        # Get the split-adjusted price data using the technical indicator function
        df = self.get_technical_indicator(symbol, function="splitadjusted")
        
        if df.empty:
            print(f"No split-adjusted data found for {symbol}")
            return
        
        # Filter data by date range
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            df.set_index('date', inplace=True)
        
        if df.empty:
            print(f"No data found for {symbol} between {start_date} and {end_date}")
            return
        
        # Create figure
        if include_volume and 'volume' in df.columns:
            # Create figure with secondary y-axis for volume
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['close'],
                mode='lines',
                name=f"{symbol} Adjusted Close",
                line=dict(color='royalblue')
            ))
            
            # Add volume as a bar chart on secondary y-axis
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker=dict(color='lightgray'),
                opacity=0.7,
                yaxis='y2'
            ))
            
            # Set up the layout with secondary y-axis
            chart_title = title if title else f"{symbol} Split-Adjusted Price with Volume"
            fig.update_layout(
                title=chart_title,
                xaxis_title='Date',
                yaxis_title='Price ($)',
                yaxis2=dict(
                    title='Volume',
                    titlefont=dict(color='gray'),
                    tickfont=dict(color='gray'),
                    overlaying='y',
                    side='right'
                ),
                template='plotly_white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
        else:
            # Simple line chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['close'],
                mode='lines',
                name=f"{symbol} Adjusted Close"
            ))
            
            # Set title
            chart_title = title if title else f"{symbol} Split-Adjusted Price"
            fig.update_layout(
                title=chart_title,
                xaxis_title='Date',
                yaxis_title='Price ($)',
                template='plotly_white'
            )
        
        # Display the plot
        fig.show()
        
    def plot_dividend_adjusted_price(self, symbol: str, start_date: str, end_date: str,
                                   title: Optional[str] = None,
                                   include_dividends: bool = True) -> None:
        """
        Plot the dividend-adjusted price history for a symbol using Plotly.
        
        Args:
            symbol (str): Stock symbol (e.g., "AAPL" or "AAPL.US")
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            title (Optional[str]): Custom chart title
            include_dividends (bool): If True, show dividend events on the chart
        """
        # Get historical prices
        price_df = self.get_historical_prices(symbol, start_date, end_date)
        
        if price_df.empty:
            print(f"No price data found for {symbol} between {start_date} and {end_date}")
            return
        
        # Ensure date column is datetime
        if 'date' in price_df.columns:
            price_df['date'] = pd.to_datetime(price_df['date'])
            price_df.set_index('date', inplace=True)
        
        # Get dividend data
        div_df = self.get_dividends_data(symbol)
        
        if div_df.empty:
            print(f"No dividend data found for {symbol}. Showing unadjusted prices.")
        else:
            # Process dividend data
            if 'date' in div_df.columns:
                div_df['date'] = pd.to_datetime(div_df['date'])
                # Filter dividends to relevant time period
                div_df = div_df[(div_df['date'] >= start_date) & (div_df['date'] <= end_date)]
                
            if not div_df.empty and 'value' in div_df.columns:
                # Create a copy of the price dataframe to avoid modifying the original
                adj_df = price_df.copy()
                
                # Calculate dividend-adjusted prices going backward in time
                adj_df['adj_close'] = adj_df['close'].copy()
                
                # Sort dividend dataframe by date in descending order
                div_df = div_df.sort_values('date', ascending=False)
                
                # For each dividend, adjust prior prices
                for _, div_row in div_df.iterrows():
                    div_date = div_row['date']
                    div_amount = div_row['value']
                    
                    # Get the closing price on the dividend date
                    try:
                        price_on_div_date = adj_df.loc[div_date, 'close']
                        # Calculate adjustment factor
                        if price_on_div_date > 0:
                            adj_factor = 1 - (div_amount / price_on_div_date)
                            # Apply adjustment to all prices before the dividend date
                            adj_df.loc[adj_df.index < div_date, 'adj_close'] *= adj_factor
                    except (KeyError, TypeError):
                        # Handle case where dividend date is not a trading day
                        print(f"Warning: Could not adjust for dividend on {div_date}")
                        continue
                
                # Create figure with adjusted close prices
                fig = go.Figure()
                
                # Add adjusted close line
                fig.add_trace(go.Scatter(
                    x=adj_df.index,
                    y=adj_df['adj_close'],
                    mode='lines',
                    name=f"{symbol} Dividend-Adjusted Close",
                    line=dict(color='green')
                ))
                
                # Add original close for comparison
                fig.add_trace(go.Scatter(
                    x=price_df.index,
                    y=price_df['close'],
                    mode='lines',
                    name=f"{symbol} Unadjusted Close",
                    line=dict(color='blue', dash='dash')
                ))
                
                # Add dividend events if requested
                if include_dividends:
                    for _, div_row in div_df.iterrows():
                        div_date = div_row['date']
                        div_amount = div_row['value']
                        
                        # Try to get the price on dividend date
                        try:
                            price_on_div = price_df.loc[div_date, 'close']
                            
                            # Add vertical line for dividend date
                            fig.add_shape(
                                type="line",
                                x0=div_date,
                                y0=0,
                                x1=div_date,
                                y1=price_on_div,
                                line=dict(color="red", width=1, dash="dot")
                            )
                            
                            # Add annotation for dividend amount
                            fig.add_annotation(
                                x=div_date,
                                y=price_on_div,
                                text=f"${div_amount:.2f}",
                                showarrow=True,
                                arrowhead=2,
                                arrowwidth=1,
                                arrowcolor="red",
                                ax=0,
                                ay=-40
                            )
                        except (KeyError, TypeError):
                            # If price on dividend date is not available, skip annotation
                            continue
                
                # Set title
                chart_title = title if title else f"{symbol} Dividend-Adjusted Price History"
                
            else:
                # No dividends found, just plot regular price
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=price_df.index,
                    y=price_df['close'],
                    mode='lines',
                    name=f"{symbol} Close"
                ))
                
                # Set title
                chart_title = title if title else f"{symbol} Price History (No Dividends)"
        
        # Update layout
        fig.update_layout(
            title=chart_title,
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Display the plot
        fig.show()
    
    def plot_total_return(self, symbol: str, start_date: str, end_date: str,
                         title: Optional[str] = None,
                         include_events: bool = True,
                         use_split_adjusted: bool = True) -> pd.DataFrame:
        """
        Plot the total return price history for a symbol, adjusting for both splits and dividends.
        
        Args:
            symbol (str): Stock symbol (e.g., "AAPL" or "AAPL.US")
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            title (Optional[str]): Custom chart title
            include_events (bool): If True, show dividend events on the chart
            use_split_adjusted (bool): If True, use split-adjusted prices before applying dividend adjustments
            
        Returns:
            pd.DataFrame: DataFrame with total return data
        """
        # Get price data - either split-adjusted or regular historical prices
        if use_split_adjusted:
            try:
                # Get split-adjusted data using technical indicator
                price_df = self.get_technical_indicator(symbol, function="splitadjusted")
                if price_df.empty:
                    print(f"No split-adjusted data found for {symbol}. Falling back to regular historical prices.")
                    price_df = self.get_historical_prices(symbol, start_date, end_date)
                else:
                    # Ensure data is within requested date range
                    if 'date' in price_df.columns:
                        price_df['date'] = pd.to_datetime(price_df['date'])
                        price_df = price_df[(price_df['date'] >= start_date) & (price_df['date'] <= end_date)]
                        if 'date' not in price_df.index.names:  # Check if date is not already the index
                            price_df.set_index('date', inplace=True)
            except Exception as e:
                print(f"Error getting split-adjusted data: {e}. Falling back to regular historical prices.")
                price_df = self.get_historical_prices(symbol, start_date, end_date)
        else:
            # Use regular historical prices
            price_df = self.get_historical_prices(symbol, start_date, end_date)
        
        if price_df.empty:
            print(f"No price data found for {symbol} between {start_date} and {end_date}")
            return pd.DataFrame()  # Return empty DataFrame
        
        # Handle various column name formats from API
        close_col = 'close' if 'close' in price_df.columns else 'adjusted_close' if 'adjusted_close' in price_df.columns else None
        if close_col is None:
            print(f"Could not find a recognized price column in the data for {symbol}")
            return price_df  # Return the original dataframe without modifications
        
        # Create a copy to avoid modifying the original
        adj_df = price_df.copy()
        
        # Get dividend data
        try:
            div_df = self.get_dividends_data(symbol)
        except Exception as e:
            print(f"Error getting dividend data: {e}")
            div_df = pd.DataFrame()  # Empty DataFrame if can't get dividend data
        
        # Initialize for plot
        if not div_df.empty and 'date' in div_df.columns:
            # Process dividend data
            div_df['date'] = pd.to_datetime(div_df['date'])
            # Filter dividends to relevant time period
            div_df = div_df[(div_df['date'] >= start_date) & (div_df['date'] <= end_date)]
            
            if not div_df.empty and 'value' in div_df.columns:
                # Calculate dividend-adjusted prices starting with split-adjusted prices
                adj_df['total_return'] = adj_df[close_col].copy()
                
                # Sort dividend dataframe by date in descending order
                div_df = div_df.sort_values('date', ascending=False)
                
                # For each dividend, adjust prior prices
                for _, div_row in div_df.iterrows():
                    div_date = div_row['date']
                    div_amount = div_row['value']
                    
                    # Get the closing price on the dividend date
                    try:
                        price_on_div_date = adj_df.loc[div_date, close_col]
                        # Calculate adjustment factor
                        if price_on_div_date > 0:
                            adj_factor = 1 - (div_amount / price_on_div_date)
                            # Apply adjustment to all prices before the dividend date
                            adj_df.loc[adj_df.index < div_date, 'total_return'] *= adj_factor
                    except (KeyError, TypeError) as e:
                        # Handle case where dividend date is not a trading day
                        print(f"Warning: Could not adjust for dividend on {div_date}: {e}")
                        continue
                
                # Set total_return column for return data
                adj_df['total_return'] = adj_df['total_return']
        else:
            # No dividends, total return is just the price
            adj_df['total_return'] = adj_df[close_col]
        
        # Return the adjusted dataframe
        return adj_df


# Main event:
if __name__ == "__main__":
    # Connect to EODHD API
    api_key = "5bbb687655d4f6.67318360"
    client = EODHDClient(api_key=api_key)  

    # Define the symbol and exchange (VIXY - Shows Split Adjustment)
    #symbol = "VIXY"
    #exchange = "BATS"
    #interval = "15m"
    #start_date = "2023-01-01"
    #end_date = "2025-01-01"
    #timezone = "UTC"

    # Define the symbol and exchange (AAPL - Shows Dividend and Split Adjustment)
    symbol = "AAPL"
    exchange = "US"
    interval = "15m"
    start_date = "2025-01-01"
    end_date = "2025-04-09"
    timezone = "UTC"
    
    # Plot total return with both split and dividend adjustments
    data_df=client.plot_total_return(symbol, start_date, end_date,title="Dividend and Split Adjusted - with Dividend Events")
    print(data_df.describe())

    # Use raw prices without split adjustment (only dividend-adjusted)
    client.plot_total_return(symbol, start_date, end_date, use_split_adjusted=False,title="Dividend Adjusted - No Split Adjustment")

    # Hide dividend event markers
    client.plot_total_return(symbol, start_date, end_date, include_events=False,title="Dividend and Split Adjusted - No Dividend Events")

    # Analyze price drops
    client.plot_closing_price(symbol, start_date, end_date, title="Closing Price",use_candlestick=True)   