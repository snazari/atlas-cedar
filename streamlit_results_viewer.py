import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime
import hashlib
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Import beta analyzer
from beta_analyzer_streamlit import StreamlitBetaAnalyzer

# Configuration
PASSWORD_HASH = "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"  # Default: "password"
# To generate a new password hash, use: hashlib.sha256("your_password".encode()).hexdigest()

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyBt5Y1JsKMS5kuvkZpRhV1CKa06SHU4z-s"

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == PASSWORD_HASH:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

def get_portfolio_data(asset_name=None):
    """Fetches portfolio data from the SQLite database."""
    if not os.path.exists('portfolio_data.db'):
        return pd.DataFrame()
    
    conn = sqlite3.connect('portfolio_data.db')
    try:
        # First check if multi-asset table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_data'")
        multi_asset_exists = cursor.fetchone() is not None
        
        if multi_asset_exists:
            # Use multi-asset table
            if asset_name:
                query = f"SELECT * FROM portfolio_data WHERE asset_name = '{asset_name}' ORDER BY timestamp ASC"
            else:
                query = "SELECT * FROM portfolio_data ORDER BY asset_name, timestamp ASC"
            df = pd.read_sql_query(query, conn)
        else:
            # Fall back to old btc_balance table
            df = pd.read_sql_query("SELECT * FROM btc_balance ORDER BY timestamp ASC", conn)
            df['asset_name'] = 'BTC-USD'  # Add asset name column for compatibility
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_available_assets():
    """Get list of available assets in the database."""
    if not os.path.exists('portfolio_data.db'):
        return []
    
    conn = sqlite3.connect('portfolio_data.db')
    try:
        cursor = conn.cursor()
        # Check if multi-asset table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_data'")
        if cursor.fetchone():
            cursor.execute("SELECT DISTINCT asset_name FROM portfolio_data ORDER BY asset_name")
            return [row[0] for row in cursor.fetchall()]
        else:
            # If only old table exists, return BTC-USD
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='btc_balance'")
            if cursor.fetchone():
                return ['BTC-USD']
        return []
    except:
        return []
    finally:
        conn.close()

def generate_ai_summary(asset_name, metrics):
    """Generate an AI-powered summary of asset performance using Google Gemini."""
    if not GEMINI_AVAILABLE:
        return "⚠️ Google Gemini library not installed. Run: pip install google-generativeai"
    
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API key not set. Set GEMINI_API_KEY environment variable."
    
    try:
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-pro')  # Updated model name
        
        # Construct prompt with metrics
        market_name = metrics.get('market_name', 'Market')
        prompt = f"""You are a financial analyst. Provide a concise one-paragraph summary (3-4 sentences) of this cryptocurrency trading algorithm's performance.

Asset: {asset_name}

Key Performance Metrics:
- Total Gain: {metrics.get('gain_percent', 'N/A'):.2f}% (First to Last Portfolio Value)
- Initial Drawdown: {metrics.get('dd_init_percent', 'N/A'):.2f}% (Maximum loss from initial value)
- Annualized Alpha: {metrics.get('alpha_percent', 'N/A')} (Excess return over {market_name})
- Beta vs {market_name}: {metrics.get('beta', 'N/A')} (Market sensitivity)
- Current Portfolio Value: ${metrics.get('current_value', 'N/A'):,.2f}
- Initial Portfolio Value: ${metrics.get('initial_value', 'N/A'):,.2f}
- Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}
- Date Range: {metrics.get('date_range', 'N/A')}
- Data Points: {metrics.get('data_points', 'N/A')}

Provide a professional analysis focusing on:
1. Overall performance (total gain and how it compares to the market)
2. Be sure to compare the total gain to the market performance
3. Risk profile (initial drawdown and what it means)
4. Alpha generation (risk-adjusted excess returns)
5. Market correlation (beta interpretation - higher/lower volatility than {market_name})
6. Key strengths or concerns

Keep it concise and actionable."""
        
        # Generate summary
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"⚠️ Error generating summary: {str(e)}"

def display_live_portfolio():
    """Renders the live portfolio dashboard with support for multiple assets."""
    st.header("Live Portfolio Monitor")

    # Get available assets
    available_assets = get_available_assets()
    
    if not available_assets:
        st.info("No portfolio data found. Please run the `gmail_integration.py` script or import data using `import_asset_data.py`")
        return
    
    # Asset selection
    col1, col2 = st.columns([3, 1])
    with col1:
        # Select all available assets by default (or first 5 if more than 5)
        default_selection = available_assets if len(available_assets) <= 5 else available_assets[:5]
            
        selected_assets = st.multiselect(
            "Select assets to display:",
            available_assets,
            default=default_selection
        )
    with col2:
        if st.button("Refresh Data"):
            st.rerun()
    
    if not selected_assets:
        st.warning("Please select at least one asset to display")
        return
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PORTFOLIO-WIDE SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown("---")
    st.markdown("## Portfolio Summary")
    
    # Calculate portfolio-wide statistics
    portfolio_stats = []
    total_current_value = 0
    total_initial_value = 0
    
    for asset in selected_assets:
        asset_df = get_portfolio_data(asset)
        if not asset_df.empty:
            latest_data = asset_df.iloc[-1]
            first_data = asset_df.iloc[0]
            
            # Use first and last portfolio values (current_value) for gain calculation
            first_portfolio_value = first_data['current_value']
            last_portfolio_value = latest_data['current_value']
            
            gain = last_portfolio_value - first_portfolio_value
            gain_percent = (gain / first_portfolio_value) * 100 if first_portfolio_value > 0 else 0
            
            portfolio_stats.append({
                'asset': asset,
                'current_value': last_portfolio_value,
                'initial_value': first_portfolio_value,
                'gain': gain,
                'gain_percent': gain_percent
            })
            
            total_current_value += last_portfolio_value
            total_initial_value += first_portfolio_value
    
    # Calculate overall portfolio metrics
    total_gain = total_current_value - total_initial_value
    total_gain_percent = (total_gain / total_initial_value) * 100 if total_initial_value > 0 else 0
    
    # Find best and worst performers
    if portfolio_stats:
        best_performer = max(portfolio_stats, key=lambda x: x['gain_percent'])
        worst_performer = min(portfolio_stats, key=lambda x: x['gain_percent'])
    
    # Display summary cards
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric(
            "Total Portfolio Value",
            f"${total_current_value:,.2f}",
            f"${total_gain:+,.2f} ({total_gain_percent:+.2f}%)"
        )
    
    with summary_col2:
        if portfolio_stats:
            st.metric(
                "Best Performer",
                best_performer['asset'],
                f"{best_performer['gain_percent']:+.2f}%"
            )
    
    with summary_col3:
        if portfolio_stats:
            st.metric(
                "Worst Performer",
                worst_performer['asset'],
                f"{worst_performer['gain_percent']:+.2f}%"
            )
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TABBED INTERFACE
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Custom CSS for larger tab font
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 40px;
            font-weight: 600;
        }
        </style>
        """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Equity Curves", "Statistics", "Raw Data", "Beta Analysis"])
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1: OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader("Individual Asset Overview")
        
        # Create columns and display metrics for all selected assets
        if selected_assets:
            metrics_cols = st.columns(len(selected_assets))
            
            for idx, asset in enumerate(selected_assets):
                asset_df = get_portfolio_data(asset)
                if not asset_df.empty:
                    with metrics_cols[idx]:
                        latest_data = asset_df.iloc[-1]
                        first_data = asset_df.iloc[0]
                        
                        # Use first and last portfolio values for gain calculation
                        first_portfolio_value = first_data['current_value']
                        last_portfolio_value = latest_data['current_value']
                        
                        total_pl = last_portfolio_value - first_portfolio_value
                        pl_percent = (total_pl / first_portfolio_value) * 100
                        
                        # Add icon based on asset
                        if "BTC" in asset:
                            asset_label = f"🟠 {asset}"
                        elif "ETH" in asset:
                            asset_label = f"🔵 {asset}"
                        elif "XRP" in asset:
                            asset_label = f"🔵 {asset}"
                        elif "SOL" in asset:
                            asset_label = f"🟣 {asset}"
                        else:
                            asset_label = f"📊 {asset}"
                            
                        st.metric(
                            asset_label,
                            f"${last_portfolio_value:,.2f}",
                            f"${total_pl:+,.2f} ({pl_percent:+.2f}%)"
                        )
                        
                        # Show fee if available
                        if 'fee' in latest_data and pd.notna(latest_data['fee']) and latest_data['fee'] > 0:
                            st.caption(f"Fee: ${latest_data['fee']:.2f}")
                        
                        last_updated = latest_data['timestamp'].strftime('%Y-%m-%d %H:%M')
                        st.caption(f"Updated: {last_updated}")
        
        # AI Summary Section - Display below asset cards
        st.markdown("---")
        st.markdown("### Key Performance Metrics")
        #st.markdown("*Click the button below each asset to generate an AI-powered performance summary*")
        
        # Create individual sections for each asset's AI summary
        for asset in selected_assets:
            asset_df = get_portfolio_data(asset)
            if not asset_df.empty:
                with st.container():
                    # Asset header
                    if "BTC" in asset:
                        st.markdown(f"#### 🟠 {asset}")
                        market_name = "BTC"
                    elif "ETH" in asset:
                        st.markdown(f"#### 🔵 {asset}")
                        market_name = "ETH"
                    elif "XRP" in asset:
                        st.markdown(f"#### 🔵 {asset}")
                        market_name = "XRP"
                    elif "SOL" in asset:
                        st.markdown(f"#### 🟣 {asset}")
                        market_name = "SOL"
                    else:
                        st.markdown(f"#### 📊 {asset}")
                        market_name = "Market"
                    
                    # Calculate metrics for AI summary
                    latest_data = asset_df.iloc[-1]
                    first_data = asset_df.iloc[0]
                    
                    # 1. Calculate gain using first and last portfolio_value (current_value)
                    first_portfolio_value = first_data['current_value']
                    last_portfolio_value = latest_data['current_value']
                    gain = last_portfolio_value - first_portfolio_value
                    gain_percent = (gain / first_portfolio_value) * 100 if first_portfolio_value > 0 else 0
                    
                    # 2. Calculate initial drawdown using first portfolio_value and minimum
                    min_portfolio_value = asset_df['current_value'].min()
                    dd_init = min_portfolio_value - first_portfolio_value
                    dd_init_percent = (dd_init / first_portfolio_value) * 100 if first_portfolio_value > 0 else 0
                    
                    date_range = f"{first_data['timestamp'].strftime('%Y-%m-%d')} to {latest_data['timestamp'].strftime('%Y-%m-%d')}"
                    
                    # Calculate Sharpe, Alpha, Beta
                    asset_df_copy = asset_df.set_index('timestamp')
                    hourly_returns = asset_df_copy['current_value'].resample('h').last().pct_change().dropna()
                    
                    sharpe_value = "N/A"
                    if len(hourly_returns) > 0 and hourly_returns.std() > 0:
                        # Use portfolio returns for Sharpe calculation
                        portfolio_returns = asset_df_copy['current_value'].pct_change().dropna()
                        
                        # Calculate periods per year based on data frequency
                        dt_index = portfolio_returns.index
                        if len(dt_index) > 1:
                            avg_seconds = (dt_index[-1] - dt_index[0]).total_seconds() / (len(dt_index) - 1)
                            periods_per_year = (365.25 * 24 * 3600) / avg_seconds
                            annualization_factor = np.sqrt(periods_per_year)
                            
                            sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns, ddof=1) if np.std(portfolio_returns, ddof=1) != 0 else 0
                            sharpe_ratio_annual = sharpe_ratio * annualization_factor
                            sharpe_value = f"{sharpe_ratio_annual:.2f}"
                    
                    # 3 & 4. Calculate annualized alpha and beta with respect to market
                    alpha_value = "N/A"
                    beta_value = "N/A"
                    alpha_percent = "N/A"
                    try:
                        df_capm = asset_df.copy()
                        df_capm = df_capm.reset_index()
                        df_capm['timestamp'] = pd.to_datetime(df_capm['timestamp'])
                        df_capm.set_index('timestamp', inplace=True)
                        df_capm.sort_index(inplace=True)
                        
                        if 'initial_value' in df_capm.columns and df_capm['initial_value'].notna().any():
                            df_capm['market_return'] = df_capm['initial_value'].pct_change()
                            df_capm['portfolio_return'] = df_capm['current_value'].pct_change()
                            df_capm.dropna(inplace=True)
                            
                            if len(df_capm) > 2:
                                X = df_capm['market_return'].values
                                y = df_capm['portfolio_return'].values
                                X_with_const = np.column_stack([np.ones(len(X)), X])
                                result = np.linalg.lstsq(X_with_const, y, rcond=None)
                                alpha, beta = result[0]
                                dt_index = df_capm.index
                                avg_seconds = (dt_index[-1] - dt_index[0]).total_seconds() / (len(dt_index) - 1)
                                periods_per_year = (365.25 * 24 * 3600) / avg_seconds
                                alpha_annualized = alpha * periods_per_year
                                alpha_value = f"{alpha_annualized:.4f}"
                                alpha_percent = f"{alpha_annualized * 100:.2f}%"
                                beta_value = f"{beta:.4f}"
                    except:
                        pass
                    
                    metrics_dict = {
                        'gain_percent': gain_percent,
                        'current_value': last_portfolio_value,
                        'initial_value': first_portfolio_value,
                        'sharpe_ratio': sharpe_value,
                        'alpha': alpha_value,
                        'alpha_percent': alpha_percent,
                        'beta': beta_value,
                        'dd_init_percent': dd_init_percent,
                        'data_points': len(asset_df),
                        'date_range': date_range,
                        'market_name': market_name
                    }
                    
                    # Display key parameters prominently
                    #st.markdown("##### Key Performance Metrics")
                    
                    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                    
                    with metric_col1:
                        st.metric(
                            "Total Gain",
                            f"{gain_percent:.2f}%",
                            delta=f"${gain:,.2f}",
                            help="Percentage gain from first to last portfolio value"
                        )
                    
                    with metric_col2:
                        st.metric(
                            "Initial Drawdown",
                            f"{dd_init_percent:.2f}%",
                            delta=f"${dd_init:,.2f}",
                            delta_color="inverse",
                            help="Maximum drawdown from initial portfolio value"
                        )
                    
                    with metric_col3:
                        st.metric(
                            "Annualized Alpha",
                            alpha_percent if alpha_percent != "N/A" else "N/A",
                            help="Risk-adjusted excess return over market (annualized)"
                        )
                    
                    with metric_col4:
                        st.metric(
                            f"Beta vs {market_name}",
                            beta_value if beta_value != "N/A" else "N/A",
                            help=f"Portfolio sensitivity to {market_name} price movements"
                        )
                    
                    with metric_col5:
                        st.metric(
                            "Sharpe Ratio",
                            sharpe_value if sharpe_value != "N/A" else "N/A",
                            help="Risk-adjusted return (annualized)"
                        )
                    
                    st.markdown("")
                    
                    # Generate AI summary button
                    if st.button(f"🤖 Generate AI Summary", key=f"ai_summary_overview_{asset}"):
                        with st.spinner("Generating AI analysis..."):
                            summary = generate_ai_summary(asset, metrics_dict)
                            # Display in a styled container
                            st.info(summary)
                    
                    st.markdown("")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2: DETAILED CHARTS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        # Global chart options
        with st.container():
            st.subheader("Chart Settings")
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                chart_type = st.selectbox("Chart Type", ["Absolute Values", "Normalized (Base 100)", "Percentage Change"])
            with col2:
                show_markers = st.checkbox("Show Markers", value=True)
            with col3:
                sampling_freq = st.selectbox("Frequency", ["Weekly", "Daily", "Hourly"], index=2)
        
        # Color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Create a separate section for each asset
        for idx, asset in enumerate(selected_assets):
            st.divider()
            
            # Add colored header based on asset
            if "BTC" in asset:
                st.markdown(f"## 🟠 {asset} - Bitcoin")
            elif "ETH" in asset:
                st.markdown(f"## 🔵 {asset} - Ethereum")
            elif "XRP" in asset:
                st.markdown(f"## 🔵 {asset} - Ripple")
            elif "SOL" in asset:
                st.markdown(f"## 🟣 {asset} - Solana")
            else:
                st.markdown(f"## 📊 {asset} Portfolio Analysis")
            
            asset_df = get_portfolio_data(asset)
            if not asset_df.empty:
                # Calculate statistics for this asset
                latest_data = asset_df.iloc[-1]
                first_data = asset_df.iloc[0]
                min_idx = asset_df['current_value'].idxmin()
                min_val = asset_df.loc[min_idx, 'current_value']
                max_idx = asset_df['current_value'].idxmax()
                max_val = asset_df.loc[max_idx, 'current_value']
                
                # Calculate hourly returns for volatility and Sharpe ratio
                asset_df_copy = asset_df.set_index('timestamp')
                hourly_returns = asset_df_copy['current_value'].resample('h').last().pct_change().dropna()
                
                # === FIRST ROW OF METRICS ===
                st.markdown("### Key Metrics")
                row1_col1, row1_col2, row1_col3 = st.columns(3)
                
                with row1_col1:
                    st.metric("Data Points", len(asset_df))
                with row1_col2:
                    date_range = f"{first_data['timestamp'].strftime('%Y-%m-%d')} to {latest_data['timestamp'].strftime('%Y-%m-%d')}"
                    st.metric("Date Range", date_range)
                with row1_col3:
                    gain = latest_data['current_value'] - first_data['current_value']
                    gain_percent = (gain / first_data['current_value']) * 100
                    st.metric("Gain", f"{gain_percent:.2f}%")
                
                # === SECOND ROW OF METRICS ===
                st.markdown("### Performance Indicators")
                row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
                
                with row2_col1:
                    dd_init = min_val - first_data['current_value']
                    dd_init_percent = (dd_init / first_data['current_value']) * 100
                    st.metric("DD from Init", f"{dd_init_percent:.2f}%")
                
                with row2_col2:
                    # Calculate Sharpe Ratio (annualized)
                    if len(hourly_returns) > 0 and hourly_returns.std() > 0:
                        mean_return = hourly_returns.mean()
                        std_return = hourly_returns.std()
                        sharpe_ratio = (mean_return / std_return) * (252 ** 0.5)
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    else:
                        st.metric("Sharpe Ratio", "N/A")
                
                with row2_col3:
                    try:
                        # Prepare data for CAPM regression
                        df_capm = asset_df.copy()
                        df_capm = df_capm.reset_index()
                        df_capm['timestamp'] = pd.to_datetime(df_capm['timestamp'])
                        df_capm.set_index('timestamp', inplace=True)
                        df_capm.sort_index(inplace=True)
                        
                        # Check if we have initial_value column (as proxy for market/benchmark)
                        if 'initial_value' in df_capm.columns and df_capm['initial_value'].notna().any():
                            # Calculate returns
                            df_capm['market_return'] = df_capm['initial_value'].pct_change()
                            df_capm['portfolio_return'] = df_capm['current_value'].pct_change()
                            df_capm.dropna(inplace=True)
                            
                            if len(df_capm) > 2:
                                # CAPM regression: portfolio_return = alpha + beta * market_return
                                X = df_capm['market_return'].values
                                y = df_capm['portfolio_return'].values
                                
                                # Add intercept term for alpha
                                X_with_const = np.column_stack([np.ones(len(X)), X])
                                
                                # OLS regression
                                result = np.linalg.lstsq(X_with_const, y, rcond=None)
                                alpha, beta = result[0]
                                
                                # Infer frequency to annualize alpha
                                dt_index = df_capm.index
                                avg_seconds = (dt_index[-1] - dt_index[0]).total_seconds() / (len(dt_index) - 1)
                                periods_per_year = (365.25 * 24 * 3600) / avg_seconds
                                alpha_annualized = alpha * periods_per_year
                                
                                st.metric("Alpha (Ann.)", f"{alpha_annualized:.4f}")
                            else:
                                st.metric("Alpha (Ann.)", "N/A")
                        else:
                            st.metric("Alpha (Ann.)", "N/A")
                    except Exception as e:
                        st.metric("Alpha (Ann.)", "N/A")
            
                with row2_col4:
                    try:
                        # Use same CAPM data from above
                        df_capm = asset_df.copy()
                        df_capm = df_capm.reset_index()
                        df_capm['timestamp'] = pd.to_datetime(df_capm['timestamp'])
                        df_capm.set_index('timestamp', inplace=True)
                        df_capm.sort_index(inplace=True)
                        
                        if 'initial_value' in df_capm.columns and df_capm['initial_value'].notna().any():
                            df_capm['market_return'] = df_capm['initial_value'].pct_change()
                            df_capm['portfolio_return'] = df_capm['current_value'].pct_change()
                            df_capm.dropna(inplace=True)
                            
                            if len(df_capm) > 2:
                                X = df_capm['market_return'].values
                                y = df_capm['portfolio_return'].values
                                X_with_const = np.column_stack([np.ones(len(X)), X])
                                result = np.linalg.lstsq(X_with_const, y, rcond=None)
                                alpha, beta = result[0]
                                
                                st.metric("Beta", f"{beta:.4f}")
                            else:
                                st.metric("Beta", "N/A")
                        else:
                            st.metric("Beta", "N/A")
                    except Exception as e:
                        st.metric("Beta", "N/A")
                
                # Create the chart for this specific asset
                fig = go.Figure()
                
                # Set timestamp as index for resampling
                asset_df = asset_df.set_index('timestamp')
                
                # Resample based on selected frequency
                if sampling_freq == "Weekly":
                    resampled_df = asset_df[['current_value']].resample('W').last()
                    resampled_df = resampled_df.fillna(method='ffill')
                elif sampling_freq == "Daily":
                    resampled_df = asset_df[['current_value']].resample('D').last()
                    resampled_df = resampled_df.fillna(method='ffill')
                else:  # Hourly - use original data
                    resampled_df = asset_df[['current_value']]
                
                # Reset index to get timestamp back as column
                resampled_df = resampled_df.reset_index()
                x_values = resampled_df['timestamp']
                
                if chart_type == "Absolute Values":
                    y_values = resampled_df['current_value']
                    y_title = "Portfolio Value (USD)"
                elif chart_type == "Normalized (Base 100)":
                    first_value = resampled_df['current_value'].iloc[0]
                    y_values = (resampled_df['current_value'] / first_value) * 100
                    y_title = "Normalized Value (Base = 100)"
                else:  # Percentage Change
                    first_value = resampled_df['current_value'].iloc[0]
                    y_values = ((resampled_df['current_value'] - first_value) / first_value) * 100
                    y_title = "Percentage Change (%)"
                
                mode = 'lines+markers' if show_markers else 'lines'
                
                # Main trace
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode=mode,
                    name=asset,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    marker=dict(size=6) if show_markers else None,
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Value: %{y:.2f}<extra></extra>'
                ))
                
                # Add min/max markers
                min_idx = y_values.idxmin()
                max_idx = y_values.idxmax()
                
                fig.add_trace(go.Scatter(
                    x=[x_values.iloc[min_idx]],
                    y=[y_values.iloc[min_idx]],
                    mode='markers+text',
                    name='Min',
                    marker=dict(color='red', size=10, symbol='triangle-down'),
                    text=[f"Min: {y_values.iloc[min_idx]:.2f}"],
                    textposition="bottom center",
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=[x_values.iloc[max_idx]],
                    y=[y_values.iloc[max_idx]],
                    mode='markers+text',
                    name='Max',
                    marker=dict(color='green', size=10, symbol='triangle-up'),
                    text=[f"Max: {y_values.iloc[max_idx]:.2f}"],
                    textposition="top center",
                    showlegend=False
                ))
                
                fig.update_layout(
                    title=f"{asset} - {chart_type} ({sampling_freq})",
                    xaxis_title="Date",
                    yaxis_title=y_title,
                    template="plotly_white",
                    hovermode='x unified',
                    height=400,
                    showlegend=False
                )
                
                # Add grid
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                
                st.plotly_chart(fig, use_container_width=True)
            
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3: STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Detailed Statistics")
        
        # Create a statistics table for all assets
        stats_data = []
        for asset in selected_assets:
            asset_df = get_portfolio_data(asset)
            if not asset_df.empty:
                latest_data = asset_df.iloc[-1]
                first_data = asset_df.iloc[0]
                
                # Use first and last portfolio values for gain calculation
                first_portfolio_value = first_data['current_value']
                last_portfolio_value = latest_data['current_value']
                
                gain = last_portfolio_value - first_portfolio_value
                gain_percent = (gain / first_portfolio_value) * 100 if first_portfolio_value > 0 else 0
                
                min_val = asset_df['current_value'].min()
                max_val = asset_df['current_value'].max()
                avg_val = asset_df['current_value'].mean()
                
                stats_data.append({
                    'Asset': asset,
                    'Current Value': f"${last_portfolio_value:,.2f}",
                    'Initial Value': f"${first_portfolio_value:,.2f}",
                    'Gain': f"${gain:+,.2f}",
                    'Gain %': f"{gain_percent:+.2f}%",
                    'Min': f"${min_val:,.2f}",
                    'Max': f"${max_val:,.2f}",
                    'Avg': f"${avg_val:,.2f}",
                    'Data Points': len(asset_df)
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4: RAW DATA
    # ═══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("Raw Data Export")
        
        for asset in selected_assets:
            with st.expander(f"📄 {asset} - View/Download Data"):
                asset_df = get_portfolio_data(asset)
                if not asset_df.empty:
                    # Show data info
                    st.info(f"Total records: {len(asset_df)}")
                    
                    # Display options
                    col1, col2 = st.columns(2)
                    with col1:
                        num_rows = st.selectbox(
                            f"Number of rows to display ({asset})",
                            [10, 25, 50, 100, "All"],
                            key=f"rows_{asset}"
                        )
                    with col2:
                        sort_order = st.selectbox(
                            f"Sort order ({asset})",
                            ["Newest First", "Oldest First"],
                            key=f"sort_{asset}"
                        )
                    
                    # Prepare display dataframe
                    display_df = asset_df.copy()
                    display_df = display_df.reset_index()
                    
                    if sort_order == "Newest First":
                        display_df = display_df.sort_values('timestamp', ascending=False)
                    else:
                        display_df = display_df.sort_values('timestamp', ascending=True)
                    
                    if num_rows != "All":
                        display_df = display_df.head(int(num_rows))
                    
                    # Format timestamp
                    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Display dataframe
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Download button
                    csv = asset_df.to_csv(index=False)
                    st.download_button(
                        label=f"⬇️ Download {asset} as CSV",
                        data=csv,
                        file_name=f"{asset}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key=f"download_{asset}"
                    )
                else:
                    st.warning(f"No data available for {asset}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 5: BETA ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab5:
        st.subheader("📉 Comprehensive Beta & Performance Analysis")
        st.markdown("*Advanced statistical analysis of portfolio performance vs market benchmark*")
        
        # Asset selection for beta analysis
        selected_asset_beta = st.selectbox(
            "Select asset for beta analysis:",
            selected_assets,
            key="beta_asset_selector"
        )
        
        if selected_asset_beta:
            asset_df = get_portfolio_data(selected_asset_beta)
            
            if not asset_df.empty and len(asset_df) >= 30:
                try:
                    # Initialize analyzer
                    analyzer = StreamlitBetaAnalyzer(
                        asset_name=selected_asset_beta,
                        confidence_level=0.95,
                        risk_free_rate=0.0
                    )
                    
                    # Load data
                    analyzer.load_data_from_db(asset_df)
                    
                    # Calculate metrics
                    with st.spinner("Calculating beta and alpha metrics..."):
                        beta_results = analyzer.calculate_beta_alpha()
                        perf_results = analyzer.calculate_performance_ratios()
                    
                    # ========================================================================
                    # PERFORMANCE COMPARISON (TOP SECTION)
                    # ========================================================================
                    st.markdown("")
                    st.markdown("## Performance Comparison")
                    st.markdown("*Strategy vs Market Total Returns*")
                    
                    # Large prominent metrics for performance comparison
                    perf_col1, perf_col2, perf_col3 = st.columns([1, 1, 1])
                    
                    with perf_col1:
                        st.metric(
                            "Strategy Total Gain",
                            f"{perf_results['strategy_gain']:.2%}",
                            help="Total return of the strategy over the period"
                        )
                    
                    with perf_col2:
                        st.metric(
                            "Market Total Gain",
                            f"{perf_results['market_gain']:.2%}",
                            help="Total return of the market over the period"
                        )
                    
                    with perf_col3:
                        # Calculate outperformance
                        outperformance = perf_results['strategy_gain'] - perf_results['market_gain']
                        st.metric(
                            "✨ Outperformance",
                            f"{outperformance:.2%}",
                            delta=f"{outperformance:.2%}",
                            help="Strategy return minus market return"
                        )
                    
                    st.markdown("---")
                    
                    # ========================================================================
                    # CORE BETA & ALPHA METRICS
                    # ========================================================================
                    st.markdown("## Core Beta & Alpha Metrics")
                    
                    # Beta metrics section
                    st.markdown("#### Beta Analysis")
                    st.metric(
                        "Beta (β)",
                        f"{beta_results['beta']:.4f}",
                        help="Sensitivity to market movements. β>1 means more volatile than market"
                    )
                    st.caption(f"95% CI: [{beta_results['beta_ci_lower']:.4f}, {beta_results['beta_ci_upper']:.4f}]")
                    
                    # Alpha metrics section
                    st.markdown("")
                    st.markdown("#### Alpha Analysis")
                    st.metric(
                        "Annualized Alpha",
                        f"{beta_results['alpha_ols_annual']:.2%}",
                        help="Risk-adjusted excess return over market (annualized)"
                    )
                    
                    st.markdown("---")
                    
                    # ========================================================================
                    # RISK-ADJUSTED PERFORMANCE RATIOS
                    # ========================================================================
                    st.markdown("## Risk-Adjusted Performance Ratios")
                    
                    ratio_col1, ratio_col2, ratio_col3, ratio_col4 = st.columns(4)
                    
                    with ratio_col1:
                        sharpe_color = "normal" if perf_results['sharpe_ratio_annual'] > 1 else "inverse"
                        st.metric(
                            "Sharpe Ratio",
                            f"{perf_results['sharpe_ratio_annual']:.4f}",
                            delta_color=sharpe_color,
                            help="Return per unit of total risk"
                        )
                    
                    with ratio_col2:
                        st.metric(
                            "Sortino Ratio",
                            f"{perf_results['sortino_ratio_annual']:.4f}",
                            help="Return per unit of downside risk"
                        )
                    
                    with ratio_col3:
                        st.metric(
                            "Treynor Ratio",
                            f"{perf_results['treynor_ratio_annual']:.2%}",
                            help="Return per unit of systematic risk"
                        )
                    
                    with ratio_col4:
                        st.metric(
                            "Information Ratio",
                            f"{perf_results['information_ratio_annual']:.4f}",
                            help="Excess return per unit of tracking error"
                        )
                    
                    st.markdown("---")
                    
                    # ========================================================================
                    # RISK METRICS
                    # ========================================================================
                    st.markdown("## Risk Metrics")
                    
                    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
                    
                    with risk_col1:
                        st.metric(
                            "Max DD (Strategy)",
                            f"{perf_results['max_drawdown_strategy']:.2%}",
                            help="Largest peak-to-trough decline for strategy"
                        )
                    
                    with risk_col2:
                        st.metric(
                            "Max DD (Market)",
                            f"{perf_results['max_drawdown_asset']:.2%}",
                            help="Largest peak-to-trough decline for market"
                        )
                    
                    with risk_col3:
                        st.metric(
                            "Calmar Ratio",
                            f"{perf_results['calmar_ratio']:.4f}",
                            help="Return / Max Drawdown"
                        )
                    
                    with risk_col4:
                        st.metric(
                            "Tracking Error",
                            f"{perf_results['tracking_error_annual']:.2%}",
                            help="Volatility of excess returns"
                        )

                    # ========================================================================
                    # VISUALIZATIONS
                    # ========================================================================
                    st.markdown("---")
                    st.markdown("### Comprehensive Dashboard")
                    
                    # Main comprehensive dashboard
                    dashboard_fig = analyzer.create_comprehensive_dashboard()
                    st.plotly_chart(dashboard_fig, use_container_width=True)
                    st.caption("📈 8-Panel comprehensive beta analysis dashboard with all key visualizations")
                    
                    # Additional detailed charts
                    st.markdown("---")
                    st.markdown("### Additional Analysis")
                    
                    detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs([
                        "Scatter Plot", 
                        "Drawdown", 
                        "Return Distribution", 
                        "Risk-Return Profile"
                    ])
                    
                    with detail_tab1:
                        scatter_fig = analyzer.create_scatter_plot()
                        st.plotly_chart(scatter_fig, use_container_width=True)
                        st.caption("Scatter plot showing the relationship between market and strategy returns with regression line")
                    
                    with detail_tab2:
                        dd_fig = analyzer.create_drawdown_plot()
                        st.plotly_chart(dd_fig, use_container_width=True)
                        st.caption("Drawdown analysis showing peak-to-trough declines over time")
                    
                    with detail_tab3:
                        dist_fig = analyzer.create_return_distribution_plot()
                        st.plotly_chart(dist_fig, use_container_width=True)
                        st.caption("Return distribution comparison between strategy and market")
                    
                    with detail_tab4:
                        risk_fig = analyzer.create_risk_return_plot()
                        st.plotly_chart(risk_fig, use_container_width=True)
                        st.caption("Risk-return profile showing annualized metrics for strategy vs market")
                    
                except Exception as e:
                    st.error(f"Error in beta analysis: {str(e)}")
                    st.info("Make sure the data has both 'initial_value' (market price) and 'current_value' (portfolio value) columns.")
            
            elif len(asset_df) < 30:
                st.warning("Insufficient data points for reliable beta analysis. Need at least 30 observations.")
            else:
                st.info("No data available for selected asset.")

def find_results_directories(base_path="."):
    """Find all directories matching the pattern 'results_*' or 'results'."""
    results_dirs = []
    
    # Look for 'results' directories and 'results_*' pattern directories
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if dir_name == "results" or dir_name.startswith("results_"):
                full_path = os.path.join(root, dir_name)
                results_dirs.append(full_path)
    
    return sorted(results_dirs)

def load_json_files_from_directory(directory):
    """Load all JSON files from a directory and return as a list of dictionaries."""
    json_files = glob.glob(os.path.join(directory, "*.json"))
    data_list = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Add metadata about the source
                data['source_file'] = os.path.basename(json_file)
                data['source_directory'] = os.path.basename(directory)
                data_list.append(data)
        except Exception as e:
            st.warning(f"Error reading {json_file}: {str(e)}")
    
    return data_list

def combine_json_data(all_data):
    """Combine all JSON data into a pandas DataFrame."""
    if not all_data:
        return pd.DataFrame()
    
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(all_data)
    
    # Reorder columns to put important ones first
    priority_columns = [
        'Symbol', 'ROI (Atlas Cedar)', 'ROI (Benchmark)', 
        'Initial Money', 'Combined Value in Quote',
        'Start Date', 'End Date', 'source_directory', 'source_file'
    ]
    
    # Get all columns
    all_columns = df.columns.tolist()
    
    # Reorder columns
    ordered_columns = []
    for col in priority_columns:
        if col in all_columns:
            ordered_columns.append(col)
            all_columns.remove(col)
    
    # Add remaining columns
    ordered_columns.extend(sorted(all_columns))
    
    # Reorder dataframe
    df = df[ordered_columns]
    
    return df

def format_dataframe(df):
    """Format the dataframe for better display."""
    if df.empty:
        return df
    
    # Format numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_columns:
        if 'ROI' in col or 'percent' in col.lower():
            df[col] = df[col].round(2)
        elif 'Price' in col or 'Money' in col or 'Value' in col:
            df[col] = df[col].round(2)
        else:
            df[col] = df[col].round(4)
    
    return df

def main():
    st.set_page_config(
        page_title="Atlas Cedar Reporting Dashboard",
        page_icon="",
        layout="wide"
    )

    st.title("Atlas Cedar Reporting Dashboard")

    if not check_password():
        st.stop()
    
    # Only show Live Portfolio for now (backtest functionality preserved but hidden)
    display_live_portfolio()
    
    # Commented out tab navigation - can be re-enabled later
    # tab1, tab2 = st.tabs(["Backtest Results", "Live Portfolio"])
    # with tab1:
    #     display_backtest_results()
    # with tab2:
    #     display_live_portfolio()

def display_backtest_results():
    # Find all results directories
    st.sidebar.header("Settings")
    base_path = st.sidebar.text_input("Base search path", value=".")
    
    results_dirs = find_results_directories(base_path)
    
    if not results_dirs:
        st.warning(f"No results directories found in '{base_path}'")
        st.info("Looking for directories named 'results' or starting with 'results_'")
        return
    
    # Sidebar - Directory selection
    st.sidebar.subheader("Found Directories")
    selected_dirs = st.sidebar.multiselect(
        "Select directories to include:",
        results_dirs,
        default=results_dirs
    )
    
    if not selected_dirs:
        st.info("Please select at least one directory from the sidebar.")
        return
    
    # Load all JSON files
    all_data = []
    for directory in selected_dirs:
        data_from_dir = load_json_files_from_directory(directory)
        all_data.extend(data_from_dir)
    
    if not all_data:
        st.warning("No JSON files found in the selected directories.")
        return
    
    # Combine into DataFrame
    df = combine_json_data(all_data)
    df = format_dataframe(df)
    
    # Display summary statistics
    st.header("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Configurations", len(df))
    
    with col2:
        if 'ROI in Quote' in df.columns:
            avg_roi = df['ROI in Quote'].mean()
            st.metric("Average ROI (%)", f"{avg_roi:.2f}")
    
    with col3:
        if 'Symbol' in df.columns:
            unique_symbols = df['Symbol'].nunique()
            st.metric("Unique Symbols", unique_symbols)
    
    with col4:
        if 'Total Trades' in df.columns:
            total_trades = df['Total Trades'].sum()
            st.metric("Total Trades", f"{total_trades:,}")
    
    # Display the data
    st.header("Results Table")
    
    # Add download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=f"trading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    df.columns = df.columns.str.replace('ROI in Quote', 'Atlas Cedar ROI')
    df.columns = df.columns.str.replace('ROI in Benchmark', 'Buy and Hold ROI')
    df.columns = df.columns.str.replace('Combined Value in Quote', 'Final Value')
    df.columns = df.columns.str.replace('Combined Value in Base', 'Combined Value (Base)')
    df.columns = df.columns.str.replace('Initial Money', 'Initial Investment')
    df.columns = df.columns.str.replace('Total Trades', 'Total Trades')
    
    # Show only important columns
    display_columns = [
        col for col in [
            'Symbol', 'Atlas Cedar ROI', 'Buy and Hold ROI',
            'Initial Investment', 'Final Value',
            'Start Date', 'End Date', 'Total Trades'
        ] if col in df.columns
    ]
    display_df = df[display_columns]
    
    # Sort by ROI if available
    if 'ROI (Atlas Cedar)' in display_df.columns:
        display_df = display_df.sort_values('ROI (Atlas Cedar)', ascending=False)
    
    # Display the dataframe
    st.dataframe(
        display_df,
        use_container_width=True,
        height=600
    )
    
    # Performance Analysis
    if 'ROI (Atlas Cedar)' in df.columns and len(df) > 1:
        st.header("Performance Analysis")
        
        # Top performers
        st.subheader("Top 10 Performers")
        top_performers = df.nlargest(10, 'ROI (Atlas Cedar)')[['Symbol', 'ROI (Atlas Cedar)', 'ROI (Benchmark)']]
        st.dataframe(top_performers, use_container_width=True)
    
    # Equity Curves Section
    st.header("Equity Curves")
    
    # Check if we have equity curve files
    equity_files = []
    for directory in selected_dirs:
        equity_pattern = os.path.join(directory, "equity_curve_*.csv")
        equity_files.extend(glob.glob(equity_pattern))
    
    if equity_files:
        # Let user select which equity curves to display
        st.subheader("Available Equity Curves")
        
        # Extract symbol names from filenames
        equity_options = {}
        for file in equity_files:
            basename = os.path.basename(file)
            # Extract symbol from filename (equity_curve_SYMBOL.csv)
            symbol_part = basename.replace("equity_curve_", "").replace(".csv", "")
            equity_options[symbol_part] = file
        
        selected_curves = st.multiselect(
            "Select equity curves to display:",
            options=list(equity_options.keys()),
            default=list(equity_options.keys())[:5]  # Default show first 5
        )
        
        if selected_curves:
            # Load and plot the selected equity curves
            
            # Create figure
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Portfolio Value Over Time", "Individual Strategy Performance (Confidential Business Information - Withheld)"),
                row_heights=[0.6, 0.4],
                shared_xaxes=True,
                specs=[[{"type": "scatter"}], [{"type": "scatter"}]]
            )
            
            # Color palette
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            for idx, symbol in enumerate(selected_curves):
                try:
                    # Load equity data
                    equity_data = pd.read_csv(equity_options[symbol])
                    equity_data['Date'] = pd.to_datetime(equity_data['Date'])
                    
                    # Plot total value
                    fig.add_trace(
                        go.Scatter(
                            x=equity_data['Date'],
                            y=equity_data['Total_Value'],
                            mode='lines',
                            name=f'{symbol}',
                            line=dict(color=colors[idx % len(colors)], width=2),
                            showlegend=True,
                            xaxis='x',
                            yaxis='y'
                        ),
                        row=1, col=1
                    )
                    
                    # Plot HK and AX values on second subplot
                    #fig.add_trace(
                    #    go.Scatter(
                    #        x=equity_data['Date'],
                    #        y=equity_data['HK_Value'],
                    #        mode='lines',
                    #        name=f'{symbol} - HK',
                    #        line=dict(color=colors[idx % len(colors)], width=1, dash='solid'),
                    #        showlegend=True
                    #    ),
                    #    row=2, col=1
                    #)
                    
                    #fig.add_trace(
                    #    go.Scatter(
                    #        x=equity_data['Date'],
                    #        y=equity_data['AX_Value'],
                    #        mode='lines',
                    #        name=f'{symbol} - AX',
                    #        line=dict(color=colors[idx % len(colors)], width=1, dash='dash'),
                    #        showlegend=True
                    #    ),
                    #    row=2, col=1
                    #)
                    
                    # Add rebalance markers
                    #rebalance_points = equity_data[equity_data['Rebalance'] == 1]
                    #if not rebalance_points.empty:
                    #    fig.add_trace(
                    #        go.Scatter(
                    #            x=rebalance_points['Date'],
                    #            y=rebalance_points['Total_Value'],
                    #            mode='markers',
                    #            name=f'{symbol} - Rebalance',
                    #            marker=dict(
                    #                color='red',
                    #                size=8,
                    #                symbol='x'
                    #            ),
                    #            showlegend=True
                    #        ),
                    #        row=1, col=1
                    #    )
                
                except Exception as e:
                    st.warning(f"Error loading equity curve for {symbol}: {str(e)}")
            
            # Update layout
            fig.update_layout(
                height=2000,
                title_text="Equity Curves Comparison",
                hovermode='x unified',
                template="plotly_white",  # Use white theme
                plot_bgcolor="white",
                paper_bgcolor="#f0f2f6",
                xaxis=dict(
                    title="Date",
                    type="date",
                    tickformat="%Y-%m-%d",
                    tickangle=-45,
                    showticklabels=True,
                    showgrid=True,
                    gridcolor="lightgray"
                ),
                xaxis2=dict(
                    title="Date",
                    type="date",
                    tickformat="%Y-%m-%d",
                    tickangle=-45,
                    showticklabels=True,
                    showgrid=True,
                    gridcolor="lightgray"
                ),
                yaxis=dict(
                    title="Portfolio Value",
                    showgrid=True,
                    gridcolor="lightgray"
                ),
                yaxis2=dict(
                    title="Strategy Value",
                    showgrid=True,
                    gridcolor="lightgray"
                ),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01,
                    bgcolor="white",
                    bordercolor="gray",
                    borderwidth=1
                )
            )
            
            # Configure x-axes to show dates properly
            # Get date range from the data
            all_dates = []
            for symbol in selected_curves:
                try:
                    equity_data = pd.read_csv(equity_options[symbol])
                    equity_data['Date'] = pd.to_datetime(equity_data['Date'])
                    all_dates.extend(equity_data['Date'].tolist())
                except:
                    pass
            
            if all_dates:
                min_date = min(all_dates)
                max_date = max(all_dates)
                
                # Update both x-axes
                for row in [1, 2]:
                    fig.update_xaxes(
                        type="date",
                        tickformat="%Y-%m-%d",
                        tickangle=-45,
                        tickmode="linear",
                        tick0=min_date,
                        dtick=86400000.0 * 30,  # Show tick every 30 days (in milliseconds)
                        showticklabels=True,
                        showgrid=True,
                        range=[min_date, max_date],
                        row=row, col=1
                    )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics for selected curves
            st.subheader("Performance Metrics")
            metrics_data = []
            
            for symbol in selected_curves:
                try:
                    equity_data = pd.read_csv(equity_options[symbol])
                    
                    initial_value = equity_data['Total_Value'].iloc[0]
                    final_value = equity_data['Total_Value'].iloc[-1]
                    max_value = equity_data['Total_Value'].max()
                    
                    # Calculate drawdown
                    running_max = equity_data['Total_Value'].expanding().max()
                    drawdown = (equity_data['Total_Value'] - running_max) / running_max * 100
                    max_drawdown = drawdown.min()
                    
                    metrics_data.append({
                        'Symbol': symbol,
                        'Initial Value': initial_value,
                        'Final Value': final_value,
                        'Total Return (%)': ((final_value - initial_value) / initial_value * 100),
                        'Max Value': max_value,
                        'Max Drawdown (%)': max_drawdown,
                        'Number of Rebalances': equity_data['Rebalance'].sum()
                    })
                
                except Exception as e:
                    st.warning(f"Error calculating metrics for {symbol}: {str(e)}")
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df = metrics_df.round(2)
                st.dataframe(metrics_df, use_container_width=True)
            
            # Periodic ROI Analysis
            st.header("Periodic ROI Analysis")
            
            # Period selection
            col1, col2 = st.columns([1, 3])
            with col1:
                period_type = st.selectbox(
                    "Select Period",
                    ["Weekly", "Monthly", "Quarterly"]
                )
            
            # Calculate periodic returns
            periodic_returns_data = []
            
            for symbol in selected_curves:
                try:
                    # Load equity data
                    equity_data = pd.read_csv(equity_options[symbol])
                    equity_data['Date'] = pd.to_datetime(equity_data['Date'])
                    equity_data = equity_data.sort_values('Date')
                    equity_data.set_index('Date', inplace=True)
                    
                    # Resample based on period type
                    if period_type == "Weekly":
                        resampled = equity_data['Total_Value'].resample('W').last()
                        period_label = "Week"
                    elif period_type == "Monthly":
                        resampled = equity_data['Total_Value'].resample('M').last()
                        period_label = "Month"
                    else:  # Quarterly
                        resampled = equity_data['Total_Value'].resample('Q').last()
                        period_label = "Quarter"
                    
                    # Calculate period returns
                    period_returns = resampled.pct_change() * 100
                    period_returns = period_returns.dropna()
                    
                    # Create a record for each period
                    for date, roi in period_returns.items():
                        periodic_returns_data.append({
                            'Symbol': symbol,
                            'Period End': date.strftime('%Y-%m-%d'),
                            f'{period_label}ly ROI (%)': round(roi, 2)
                        })
                
                except Exception as e:
                    st.warning(f"Error calculating periodic returns for {symbol}: {str(e)}")
            
            if periodic_returns_data:
                # Convert to DataFrame
                periodic_df = pd.DataFrame(periodic_returns_data)
                
                # Pivot table for better visualization
                pivot_df = periodic_df.pivot(
                    index='Period End',
                    columns='Symbol',
                    values=f'{period_label}ly ROI (%)'
                )
                
                # Add average column
                pivot_df['Average'] = pivot_df.mean(axis=1).round(2)
                
                # Sort by date (newest first)
                pivot_df = pivot_df.sort_index(ascending=False)
                
                # Display the table
                st.subheader(f"{period_type} Returns by Symbol")
                
                # Add color coding
                def color_negative_red(val):
                    """Color negative values red and positive values green"""
                    try:
                        color = 'color: red' if val < 0 else 'color: green'
                        return color
                    except:
                        return ''
                
                styled_df = pivot_df.style.applymap(color_negative_red)
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                # Summary statistics
                st.subheader(f"{period_type} Return Statistics")
                summary_stats = []
                
                for col in pivot_df.columns:
                    if col != 'Average':
                        stats = {
                            'Symbol': col,
                            'Mean Return (%)': pivot_df[col].mean().round(2),
                            'Std Dev (%)': pivot_df[col].std().round(2),
                            'Best Period (%)': pivot_df[col].max().round(2),
                            'Worst Period (%)': pivot_df[col].min().round(2),
                            'Positive Periods': (pivot_df[col] > 0).sum(),
                            'Negative Periods': (pivot_df[col] < 0).sum()
                        }
                        summary_stats.append(stats)
                
                summary_df = pd.DataFrame(summary_stats)
                st.dataframe(summary_df, use_container_width=True)
                
                # Download button for periodic returns
                csv_periodic = pivot_df.to_csv()
                st.download_button(
                    label=f"Download {period_type} Returns as CSV",
                    data=csv_periodic,
                    file_name=f"{period_type.lower()}_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    else:
        st.info("No equity curve files found. Run the backtest with the updated code to generate equity curve data.")


if __name__ == "__main__":
    main() 