#!/usr/bin/env python3
"""
Enhanced Beta Analysis Script v3.0 - Advanced Performance Metrics & Visual Enhancements

Building upon v2.0, this version adds:
- Comprehensive alpha calculations (regular & Jensen's)
- Additional performance ratios (Treynor, Information Ratio)
- Rich terminal output with colored tables and progress bars
- Organized results folder structure
- Enhanced logging capabilities
- All metrics calculated with Rf=0 as specified

Key improvements:
- Clear distinction between regular and Jensen's alpha
- Visual enhancement using the rich library
- Organized output structure in results/ folder
- Comprehensive logging for debugging

Last Update: 20-AUG-2025
- Running for meeting with Ambush

Author:
Sam Nazari, Ph.D.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy import optimize
import warnings
import os
import logging
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.layout import Layout
from rich import box
from rich.text import Text
from rich.columns import Columns
import time

warnings.filterwarnings('ignore')

# Initialize rich console
console = Console()

# Asset being analyzed
ASSET_NAME = 'BTC'
MARKET_PROXY = 'BTC'

class EnhancedBetaAnalyzer:
    """
    Enhanced Beta Analysis class with comprehensive performance metrics and visual output.
    """

    def __init__(self, confidence_level=0.95, rolling_window=252, risk_free_rate=0.0, data_frequency='daily'):
        """
        Initialize the Enhanced Beta Analyzer.

        Args:
            confidence_level (float): Confidence level for statistical tests (default: 0.95)
            rolling_window (int): Window size for rolling calculations (default: 252 trading days)
            risk_free_rate (float): Risk-free rate for calculations (default: 0.0 as requested)
            data_frequency (str): Frequency of data ('5min', 'hourly', 'daily')
        """
        self.confidence_level = confidence_level
        self.rolling_window = rolling_window
        self.risk_free_rate = risk_free_rate  # Set to 0 as requested
        self.data_frequency = data_frequency
        self.data = None
        self.results = {}
        self.logger = None

        # Set annualization factors based on frequency
        if data_frequency == '5min':
            self.periods_per_day = 288  # 24 hours * 60 minutes / 5 minutes
            self.periods_per_year = 365 * 288  # 105,120 for 24/7 crypto markets
        elif data_frequency == '15min':
            self.periods_per_day = 96
            self.periods_per_year = 365*96
        elif data_frequency == 'hourly':
            self.periods_per_day = 24
            self.periods_per_year = 365 * 24  # 8,760 for 24/7 crypto markets
        elif data_frequency == 'daily':
            self.periods_per_day = 1
            self.periods_per_year = 365  # 365 for crypto (trades every day)
        else:
            raise ValueError(f"Unknown data frequency: {data_frequency}. Use '5min', 'hourly', or 'daily'")

        self.annualization_factor = np.sqrt(self.periods_per_year)

        # Create results directory structure
        self._create_results_structure()

        # Initialize logging
        self._setup_logging()

        # Display initialization
        window_in_days = rolling_window / self.periods_per_day
        console.print(Panel.fit(
            f"[bold cyan]Enhanced Beta Analyzer v3.0[/bold cyan]\n"
            f"[green]Asset:[/green] {ASSET_NAME}\n"
            f"[green]Market Proxy:[/green] {MARKET_PROXY}\n"
            f"[green]Data Frequency:[/green] {data_frequency}\n"
            f"[green]Rolling Window:[/green] {rolling_window} periods (~{window_in_days:.1f} days)\n"
            f"[green]Confidence Level:[/green] {confidence_level*100:.1f}%\n"
            f"[green]Risk-Free Rate:[/green] {risk_free_rate*100:.1f}%\n"
            f"[green]Annualization Factor:[/green] {self.periods_per_year:,} periods/year",
            title="📊 Initialization",
            border_style="cyan"
        ))

    def _create_results_structure(self):
        """Create organized results folder structure."""
        self.results_dir = Path(f"./{ASSET_NAME}")
        self.csv_dir = self.results_dir / 'csv'
        self.html_dir = self.results_dir / 'html'
        self.logs_dir = self.results_dir / 'logs'

        # Create directories
        for dir_path in [self.csv_dir, self.html_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_filename = self.logs_dir / f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger('BetaAnalyzer')
        self.logger.info(f"Enhanced Beta Analyzer v3.0 initialized")
        self.logger.info(f"Results will be saved to: {self.results_dir}")

    def load_and_validate_data(self, file_path, validate_stationarity=True):
        """
        Load data with comprehensive validation and progress display.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:

            # Create main task
            task = progress.add_task("[cyan]Loading and validating data...", total=6)

            # Load the CSV file
            df = pd.read_csv(file_path)
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            progress.update(task, advance=1, description="[cyan]Data loaded...")

            # Validate required columns
            required_columns = ['date', 'close', 'results']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            progress.update(task, advance=1, description="[cyan]Columns validated...")

            # Convert and validate date column
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            progress.update(task, advance=1, description="[cyan]Dates processed...")

            # Check for data quality issues
            self._validate_data_quality(df)
            progress.update(task, advance=1, description="[cyan]Quality checked...")

            # Calculate returns
            df[f'{ASSET_NAME}_returns'] = df['close'].pct_change()
            df['strategy_returns'] = df['results'].pct_change()

            # Log returns for comparison
            df[f'{ASSET_NAME}_log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['strategy_log_returns'] = np.log(df['results'] / df['results'].shift(1))

            # Remove NaN values
            df = df.dropna()
            progress.update(task, advance=1, description="[cyan]Returns calculated...")

            # Stationarity testing if requested
            if validate_stationarity:
                self._test_stationarity(df)
            progress.update(task, advance=1, description="[cyan]Validation complete!")

        # Display data summary
        summary_table = Table(title="Data Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        # Calculate time span
        time_span = df['date'].max() - df['date'].min()
        days_span = time_span.days

        summary_table.add_row("Date Range", f"{df['date'].min().strftime('%Y-%m-%d %H:%M')} to {df['date'].max().strftime('%Y-%m-%d %H:%M')}")
        summary_table.add_row("Total Periods", f"{len(df):,}")
        summary_table.add_row("Data Frequency", f"{self.data_frequency}")
        summary_table.add_row("Time Span", f"{days_span} days")
        summary_table.add_row(f"{ASSET_NAME} Price Range", f"${df['close'].min():.2f} - ${df['close'].max():.2f}")
        summary_table.add_row("Strategy Value Range", f"${df['results'].min():.2f} - ${df['results'].max():.2f}")

        console.print(summary_table)

        self.data = df
        self.logger.info(f"Data validation complete. Final shape: {df.shape}")
        return df

    def _validate_data_quality(self, df):
        """Comprehensive data quality validation."""
        quality_issues = []

        # Check for missing values
        missing_data = df[['close', 'results']].isnull().sum()
        if missing_data.any():
            quality_issues.append(f"Missing values: {missing_data.to_dict()}")

        # Check for zero or negative values
        zero_negative = (df[['close', 'results']] <= 0).sum()
        if zero_negative.any():
            quality_issues.append(f"Zero/negative values: {zero_negative.to_dict()}")

        # Check for extreme outliers
        for col in ['close', 'results']:
            values = df[col].dropna()
            z_scores = np.abs(stats.zscore(values))
            extreme_outliers = (z_scores > 10).sum()
            if extreme_outliers > 0:
                quality_issues.append(f"{extreme_outliers} extreme outliers in {col}")

        # Check for data gaps
        df['date_diff'] = df['date'].diff().dt.days
        gaps = df[df['date_diff'] > 7]['date_diff']
        if len(gaps) > 0:
            quality_issues.append(f"{len(gaps)} time gaps detected (max: {gaps.max()} days)")

        if quality_issues:
            console.print("[yellow]Data Quality Issues:[/yellow]")
            for issue in quality_issues:
                console.print(f"  ⚠️  {issue}")
                self.logger.warning(f"Data quality issue: {issue}")
        else:
            console.print("[green]✓ Data quality checks passed[/green]")

    def _test_stationarity(self, df):
        """Test for stationarity in return series."""
        try:
            from statsmodels.tsa.stattools import adfuller

            stationarity_table = Table(title="Stationarity Test (ADF)", box=box.SIMPLE)
            stationarity_table.add_column("Series", style="cyan")
            stationarity_table.add_column("P-Value", style="yellow")
            stationarity_table.add_column("Status", style="green")

            for col in [f'{ASSET_NAME}_returns', 'strategy_returns']:
                if col in df.columns:
                    series = df[col].dropna()
                    adf_result = adfuller(series)
                    p_value = adf_result[1]

                    status = "✓ Stationary" if p_value < 0.05 else "✗ Non-stationary"
                    color = "green" if p_value < 0.05 else "red"

                    stationarity_table.add_row(
                        col,
                        f"{p_value:.6f}",
                        f"[{color}]{status}[/{color}]"
                    )

            console.print(stationarity_table)

        except ImportError:
            console.print("[yellow]⚠️  statsmodels not available - skipping stationarity tests[/yellow]")

    def calculate_comprehensive_beta(self):
        """
        Calculate beta using multiple methodologies with enhanced alpha calculations.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_and_validate_data() first.")

        console.print("\n[bold cyan]Calculating Comprehensive Beta & Alpha Metrics...[/bold cyan]")

        strategy_ret = self.data['strategy_returns'].dropna()
        market_ret = self.data[f'{ASSET_NAME}_returns'].dropna()

        # Align the series
        aligned_data = pd.DataFrame({
            'strategy': strategy_ret,
            'market': market_ret
        }).dropna()

        strategy_ret = aligned_data['strategy']
        market_ret = aligned_data['market']

        if len(aligned_data) < 30:
            raise ValueError("Insufficient data points for reliable beta calculation")

        # Method 1: Classical Covariance/Variance approach
        covariance = np.cov(strategy_ret, market_ret)[0, 1]
        market_variance = np.var(market_ret, ddof=1)
        beta_classical = covariance / market_variance if market_variance != 0 else np.nan

        # Method 2: OLS regression
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

        # Standard errors and confidence intervals
        n = len(aligned_data)
        residuals = strategy_ret - (alpha_ols + beta_ols * market_ret)
        mse = np.sum(residuals**2) / (n - 2)

        market_ret_centered = market_ret - np.mean(market_ret)
        se_beta = np.sqrt(mse / np.sum(market_ret_centered**2))

        t_critical = stats.t.ppf((1 + self.confidence_level) / 2, n - 2)
        beta_ci_lower = beta_ols - t_critical * se_beta
        beta_ci_upper = beta_ols + t_critical * se_beta

        # ENHANCED ALPHA CALCULATIONS

        # 1. Regular Alpha (simple excess return)
        strategy_mean_return = np.mean(strategy_ret)
        market_mean_return = np.mean(market_ret)
        regular_alpha = strategy_mean_return - market_mean_return

        # 2. Jensen's Alpha (CAPM-based with Rf=0)
        # Jensen's α = Rs - [Rf + β(Rm - Rf)]
        # With Rf=0: Jensen's α = Rs - β*Rm
        jensens_alpha = strategy_mean_return - (self.risk_free_rate + beta_ols * (market_mean_return - self.risk_free_rate))

        # Annualized alphas
        regular_alpha_annual = regular_alpha * self.periods_per_year
        jensens_alpha_annual = jensens_alpha * self.periods_per_year
        alpha_ols_annual = alpha_ols * self.periods_per_year

        # Statistical tests
        t_stat_beta_1 = (beta_ols - 1) / se_beta
        p_value_beta_1 = 2 * (1 - stats.t.cdf(abs(t_stat_beta_1), n - 2))

        # Durbin-Watson test
        dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)

        self.results['beta_analysis'] = {
            'beta_classical': beta_classical,
            'beta_ols': beta_ols,
            'alpha_ols': alpha_ols,
            'alpha_ols_annual': alpha_ols_annual,
            'regular_alpha': regular_alpha,
            'regular_alpha_annual': regular_alpha_annual,
            'jensens_alpha': jensens_alpha,
            'jensens_alpha_annual': jensens_alpha_annual,
            'correlation': correlation,
            'r_squared': r_squared,
            'standard_error_beta': se_beta,
            'beta_ci_lower': beta_ci_lower,
            'beta_ci_upper': beta_ci_upper,
            't_stat_beta_1': t_stat_beta_1,
            'p_value_beta_1': p_value_beta_1,
            'durbin_watson': dw_stat,
            'n_observations': n,
            'covariance': covariance,
            'market_variance': market_variance,
            'strategy_mean_return': strategy_mean_return,
            'market_mean_return': market_mean_return
        }

        # Display results in a beautiful table
        self._display_beta_results()

        return self.results['beta_analysis']

    def _display_beta_results(self):
        """Display beta and alpha results in a visually enhanced table."""
        beta_stats = self.results['beta_analysis']

        # Create main metrics table
        metrics_table = Table(title="📊 Core Beta & Alpha Metrics", box=box.DOUBLE_EDGE)
        metrics_table.add_column("Metric", style="cyan", width=30)
        metrics_table.add_column("Value", style="green", width=20)
        metrics_table.add_column("Annualized", style="yellow", width=20)

        # Beta metrics
        metrics_table.add_row(
            "Beta (β)",
            f"{beta_stats['beta_ols']:.6f}",
            "-"
        )
        metrics_table.add_row(
            "95% CI for Beta",
            f"[{beta_stats['beta_ci_lower']:.4f}, {beta_stats['beta_ci_upper']:.4f}]",
            "-"
        )

        # Alpha metrics
        metrics_table.add_row(
            "Regular Alpha",
            f"{beta_stats['regular_alpha']:.6f}",
            f"{beta_stats['regular_alpha_annual']:.2%}"
        )
        metrics_table.add_row(
            "Jensen's Alpha (CAPM)",
            f"{beta_stats['jensens_alpha']:.6f}",
            f"{beta_stats['jensens_alpha_annual']:.2%}"
        )
        metrics_table.add_row(
            "Regression Alpha (α)",
            f"{beta_stats['alpha_ols']:.6f}",
            f"{beta_stats['alpha_ols_annual']:.2%}"
        )

        # Statistical measures
        metrics_table.add_row(
            "R-squared",
            f"{beta_stats['r_squared']:.6f}",
            "-"
        )
        metrics_table.add_row(
            "Correlation",
            f"{beta_stats['correlation']:.6f}",
            "-"
        )

        console.print(metrics_table)

        # Statistical significance panel
        sig_text = Text()
        if beta_stats['p_value_beta_1'] < 0.05:
            sig_text.append("✓ Beta significantly different from 1.0", style="green")
        else:
            sig_text.append("✗ Beta not significantly different from 1.0", style="red")
        sig_text.append(f"\n(p-value: {beta_stats['p_value_beta_1']:.6f})")

        console.print(Panel(sig_text, title="Statistical Significance", border_style="blue"))

    def calculate_performance_ratios(self):
        """
        Calculate comprehensive performance ratios including Sharpe, Sortino, Treynor, and Information Ratio.
        All calculated with Rf=0 as requested.
        """
        console.print("\n[bold cyan]Calculating Performance Ratios (Rf=0)...[/bold cyan]")

        strategy_ret = self.data['strategy_returns'].dropna()
        market_ret = self.data[f'{ASSET_NAME}_returns'].dropna()
        beta = self.results['beta_analysis']['beta_ols']

        # Sharpe Ratio (with Rf=0)
        sharpe_ratio = np.mean(strategy_ret) / np.std(strategy_ret, ddof=1) if np.std(strategy_ret) != 0 else 0
        sharpe_ratio_annual = sharpe_ratio * self.annualization_factor

        # Sortino Ratio (with Rf=0)
        downside_returns = strategy_ret[strategy_ret < self.risk_free_rate]
        downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0
        sortino_ratio = np.mean(strategy_ret - self.risk_free_rate) / downside_std if downside_std != 0 else np.inf
        sortino_ratio_annual = sortino_ratio * self.annualization_factor

        # Treynor Ratio (with Rf=0)
        # Treynor = (Rp - Rf) / β
        treynor_ratio = np.mean(strategy_ret - self.risk_free_rate) / beta if beta != 0 else np.inf
        treynor_ratio_annual = treynor_ratio * self.periods_per_year

        # Information Ratio
        # IR = (Rp - Rb) / Tracking Error
        excess_returns = strategy_ret - market_ret
        tracking_error = np.std(excess_returns, ddof=1)
        information_ratio = np.mean(excess_returns) / tracking_error if tracking_error != 0 else 0
        information_ratio_annual = information_ratio * self.annualization_factor

        # ENHANCED DRAWDOWN CALCULATIONS

        # 1. Maximum Drawdown for Strategy (Rhino)
        strategy_values = self.data['results']
        strategy_cumulative = strategy_values / strategy_values.iloc[0]  # Normalized
        strategy_running_max = strategy_cumulative.expanding().max()
        strategy_drawdown = (strategy_cumulative - strategy_running_max) / strategy_running_max
        max_drawdown_strategy = strategy_drawdown.min()

        # 2. Maximum Drawdown for Asset (XRP)
        asset_prices = self.data['close']
        asset_cumulative = asset_prices / asset_prices.iloc[0]  # Normalized
        asset_running_max = asset_cumulative.expanding().max()
        asset_drawdown = (asset_cumulative - asset_running_max) / asset_running_max
        max_drawdown_asset = asset_drawdown.min()

        # 3. Initial Drawdown calculations (as per user's formula)
        initial_value_strategy = strategy_values.iloc[0]
        initial_value_asset = asset_prices.iloc[0]

        # Initial Drawdown for Strategy (Rhino)
        dd_init_strategy = np.max(initial_value_strategy - strategy_values) / initial_value_strategy

        # Initial Drawdown for Asset (XRP)
        dd_init_asset = np.max(initial_value_asset - asset_prices) / initial_value_asset

        # 4. Total Gains
        strategy_gain = (strategy_values.iloc[-1] - initial_value_strategy) / initial_value_strategy
        market_gain = (asset_prices.iloc[-1] - initial_value_asset) / initial_value_asset

        # 5. Down Percentage (% of time below initial value)
        down_percentage_strategy = (strategy_values < initial_value_strategy).sum() / len(strategy_values) * 100
        down_percentage_asset = (asset_prices < initial_value_asset).sum() / len(asset_prices) * 100

        # 6.(1-np.sum(List_Tot_Dollars>=5000)/np.shape(List_Tot_Dollars)[0])
        down_percentage_new_strat = strategy_drawdown

        # Calmar Ratio
        annual_return = np.mean(strategy_ret) * self.periods_per_year
        calmar_ratio = annual_return / abs(max_drawdown_strategy) if max_drawdown_strategy != 0 else np.inf

        # Value at Risk (VaR) and CVaR
        confidence_levels = [0.95, 0.99]
        var_cvar_results = {}

        for conf in confidence_levels:
            var_hist = np.percentile(strategy_ret, (1 - conf) * 100)
            cvar_hist = strategy_ret[strategy_ret <= var_hist].mean()

            var_cvar_results[f'var_{int(conf*100)}'] = var_hist
            var_cvar_results[f'cvar_{int(conf*100)}'] = cvar_hist

        # Skewness and Kurtosis
        skewness = stats.skew(strategy_ret)
        kurtosis = stats.kurtosis(strategy_ret)

        self.results['performance_ratios'] = {
            'sharpe_ratio': sharpe_ratio,
            'sharpe_ratio_annual': sharpe_ratio_annual,
            'sortino_ratio': sortino_ratio,
            'sortino_ratio_annual': sortino_ratio_annual,
            'treynor_ratio': treynor_ratio,
            'treynor_ratio_annual': treynor_ratio_annual,
            'information_ratio': information_ratio,
            'information_ratio_annual': information_ratio_annual,
            'calmar_ratio': calmar_ratio,
            'max_drawdown_strategy': max_drawdown_strategy,
            'max_drawdown_asset': max_drawdown_asset,
            'initial_drawdown_strategy': dd_init_strategy,
            'initial_drawdown_asset': dd_init_asset,
            'strategy_gain': strategy_gain,
            'market_gain': market_gain,
            'down_percentage_strategy': down_percentage_strategy,
            'down_percentage_asset': down_percentage_asset,
            'tracking_error': tracking_error,
            'tracking_error_annual': tracking_error * self.annualization_factor,
            'skewness': skewness,
            'kurtosis': kurtosis,
            **var_cvar_results
        }

        # Display performance ratios
        self._display_performance_ratios()

        return self.results['performance_ratios']

    def _display_performance_ratios(self):
        """Display performance ratios in visually enhanced tables."""
        ratios = self.results['performance_ratios']

        # Risk-Adjusted Return Ratios
        ratio_table = Table(title="📈 Risk-Adjusted Performance Ratios (Rf=0)", box=box.DOUBLE_EDGE)
        ratio_table.add_column("Ratio", style="cyan", width=25)
        ratio_table.add_column("Daily", style="yellow", width=15)
        ratio_table.add_column("Annualized", style="green", width=15)
        ratio_table.add_column("Interpretation", style="white", width=30)

        # Sharpe Ratio
        sharpe_color = "green" if ratios['sharpe_ratio_annual'] > 1 else "yellow" if ratios['sharpe_ratio_annual'] > 0.5 else "red"
        ratio_table.add_row(
            "Sharpe Ratio",
            f"{ratios['sharpe_ratio']:.4f}",
            f"[{sharpe_color}]{ratios['sharpe_ratio_annual']:.4f}[/{sharpe_color}]",
            self._interpret_sharpe(ratios['sharpe_ratio_annual'])
        )

        # Sortino Ratio
        sortino_color = "green" if ratios['sortino_ratio_annual'] > 2 else "yellow" if ratios['sortino_ratio_annual'] > 1 else "red"
        ratio_table.add_row(
            "Sortino Ratio",
            f"{ratios['sortino_ratio']:.4f}",
            f"[{sortino_color}]{ratios['sortino_ratio_annual']:.4f}[/{sortino_color}]",
            self._interpret_sortino(ratios['sortino_ratio_annual'])
        )

        # Treynor Ratio
        ratio_table.add_row(
            "Treynor Ratio",
            f"{ratios['treynor_ratio']:.6f}",
            f"{ratios['treynor_ratio_annual']:.2%}",
            "Return per unit of systematic risk"
        )

        # Information Ratio
        ir_color = "green" if ratios['information_ratio_annual'] > 0.5 else "yellow" if ratios['information_ratio_annual'] > 0 else "red"
        ratio_table.add_row(
            "Information Ratio",
            f"{ratios['information_ratio']:.4f}",
            f"[{ir_color}]{ratios['information_ratio_annual']:.4f}[/{ir_color}]",
            self._interpret_ir(ratios['information_ratio_annual'])
        )

        console.print(ratio_table)

        # Risk Metrics Table
        risk_table = Table(title="⚠️ Risk Metrics", box=box.SIMPLE_HEAVY)
        risk_table.add_column("Metric", style="cyan", width=25)
        risk_table.add_column("Value", style="yellow", width=20)

        risk_table.add_row("Max Drawdown (Strategy)", f"{ratios['max_drawdown_strategy']:.2%}")
        risk_table.add_row("Max Drawdown (Asset)", f"{ratios['max_drawdown_asset']:.2%}")
        risk_table.add_row("Initial Drawdown (Strategy)", f"{ratios['initial_drawdown_strategy']:.2%}")
        risk_table.add_row("Initial Drawdown (Asset)", f"{ratios['initial_drawdown_asset']:.2%}")
        risk_table.add_row("Calmar Ratio", f"{ratios['calmar_ratio']:.4f}")
        risk_table.add_row("Tracking Error (Annual)", f"{ratios['tracking_error_annual']:.2%}")
        risk_table.add_row("VaR 95%", f"{ratios['var_95']:.4%}")
        risk_table.add_row("CVaR 95%", f"{ratios['cvar_95']:.4%}")
        risk_table.add_row("Skewness", f"{ratios['skewness']:.4f}")
        risk_table.add_row("Kurtosis", f"{ratios['kurtosis']:.4f}")

        console.print(risk_table)

        # Performance Summary Table
        perf_table = Table(title="📈 Performance Summary", box=box.SIMPLE_HEAVY)
        perf_table.add_column("Metric", style="cyan", width=25)
        perf_table.add_column("Strategy", style="green", width=20)
        perf_table.add_column("Asset", style="yellow", width=20)

        perf_table.add_row("Total Gain", f"{ratios['strategy_gain']:.2%}", f"{ratios['market_gain']:.2%}")
        perf_table.add_row("Down Percentage", f"{ratios['down_percentage_strategy']:.1f}%", f"{ratios['down_percentage_asset']:.1f}%")

        console.print(perf_table)

    def _interpret_sharpe(self, sharpe):
        """Interpret Sharpe ratio value."""
        if sharpe < 0:
            return "Poor (negative returns)"
        elif sharpe < 0.5:
            return "Suboptimal"
        elif sharpe < 1:
            return "Acceptable"
        elif sharpe < 2:
            return "Good"
        else:
            return "Excellent"

    def _interpret_sortino(self, sortino):
        """Interpret Sortino ratio value."""
        if sortino < 0:
            return "Poor (negative returns)"
        elif sortino < 1:
            return "Suboptimal"
        elif sortino < 2:
            return "Good"
        else:
            return "Excellent"

    def _interpret_ir(self, ir):
        """Interpret Information ratio value."""
        if ir < -0.5:
            return "Poor (underperforming)"
        elif ir < 0:
            return "Below benchmark"
        elif ir < 0.5:
            return "Modest outperformance"
        elif ir < 1:
            return "Good skill"
        else:
            return "Exceptional skill"

    def calculate_weekly_metrics(self):
        """
        Calculate performance metrics on weekly aggregated data.
        This helps reduce noise from high-frequency data and provides more stable metrics.
        """
        console.print("\n[bold cyan]Calculating Weekly Metrics...[/bold cyan]")

        # Create a copy with date index
        weekly_df = self.data.copy()
        weekly_df.set_index('date', inplace=True)

        # Aggregate to weekly data
        # For prices/values: take the last value of each week
        # For returns: we'll recalculate from weekly prices
        weekly_prices = weekly_df[['close', 'results']].resample('W').last()

        # Calculate weekly returns
        weekly_returns = weekly_prices.pct_change().dropna()
        weekly_returns.columns = [f'{ASSET_NAME}_weekly_returns', 'strategy_weekly_returns']

        # Also calculate log returns for comparison
        weekly_log_returns = np.log(weekly_prices / weekly_prices.shift(1)).dropna()
        weekly_log_returns.columns = [f'{ASSET_NAME}_weekly_log_returns', 'strategy_weekly_log_returns']

        # Extract series
        strategy_weekly_ret = weekly_returns['strategy_weekly_returns']
        market_weekly_ret = weekly_returns[f'{ASSET_NAME}_weekly_returns']

        # Ensure we have enough data
        if len(weekly_returns) < 10:
            console.print("[yellow]⚠️  Insufficient weekly data points for reliable metrics[/yellow]")
            return None

        # Weekly Beta (using covariance method)
        covariance = np.cov(strategy_weekly_ret, market_weekly_ret)[0, 1]
        variance = np.var(market_weekly_ret, ddof=1)
        weekly_beta = covariance / variance if variance != 0 else np.nan

        # Weekly Alpha (regular and Jensen's)
        strategy_mean_weekly = np.mean(strategy_weekly_ret)
        market_mean_weekly = np.mean(market_weekly_ret)

        # Regular weekly alpha
        regular_weekly_alpha = strategy_mean_weekly - market_mean_weekly
        regular_weekly_alpha_annual = regular_weekly_alpha * 52

        # Jensen's weekly alpha (with Rf=0)
        jensens_weekly_alpha = strategy_mean_weekly - (self.risk_free_rate + weekly_beta * (market_mean_weekly - self.risk_free_rate))
        jensens_weekly_alpha_annual = jensens_weekly_alpha * 52

        # Weekly Sharpe Ratio (52 weeks per year)
        weekly_sharpe = np.mean(strategy_weekly_ret) / np.std(strategy_weekly_ret, ddof=1) if np.std(strategy_weekly_ret) != 0 else 0
        weekly_sharpe_annual = weekly_sharpe * np.sqrt(52)

        # Weekly Sortino Ratio
        weekly_downside = strategy_weekly_ret[strategy_weekly_ret < self.risk_free_rate]
        weekly_downside_std = np.std(weekly_downside, ddof=1) if len(weekly_downside) > 1 else 0
        weekly_sortino = np.mean(strategy_weekly_ret - self.risk_free_rate) / weekly_downside_std if weekly_downside_std != 0 else np.inf
        weekly_sortino_annual = weekly_sortino * np.sqrt(52)

        # Weekly Information Ratio
        weekly_excess_returns = strategy_weekly_ret - market_weekly_ret
        weekly_tracking_error = np.std(weekly_excess_returns, ddof=1)
        weekly_information_ratio = np.mean(weekly_excess_returns) / weekly_tracking_error if weekly_tracking_error != 0 else 0
        weekly_information_ratio_annual = weekly_information_ratio * np.sqrt(52)

        # Weekly Correlation and R-squared
        weekly_correlation = np.corrcoef(strategy_weekly_ret, market_weekly_ret)[0, 1]
        weekly_r_squared = weekly_correlation ** 2

        # Weekly Volatility
        weekly_vol_strategy = np.std(strategy_weekly_ret, ddof=1)
        weekly_vol_market = np.std(market_weekly_ret, ddof=1)
        weekly_vol_strategy_annual = weekly_vol_strategy * np.sqrt(52)
        weekly_vol_market_annual = weekly_vol_market * np.sqrt(52)

        # Weekly Drawdown Analysis
        weekly_cumulative = (1 + strategy_weekly_ret).cumprod()
        weekly_running_max = weekly_cumulative.expanding().max()
        weekly_drawdown = (weekly_cumulative - weekly_running_max) / weekly_running_max
        weekly_max_drawdown = weekly_drawdown.min()

        # Weekly VaR and CVaR
        weekly_var_95 = np.percentile(strategy_weekly_ret, 5)
        weekly_cvar_95 = strategy_weekly_ret[strategy_weekly_ret <= weekly_var_95].mean()

        # Store results
        self.results['weekly_metrics'] = {
            'n_weeks': len(weekly_returns),
            'date_range': f"{weekly_returns.index[0].strftime('%Y-%m-%d')} to {weekly_returns.index[-1].strftime('%Y-%m-%d')}",

            # Beta and Alpha
            'weekly_beta': weekly_beta,
            'regular_weekly_alpha': regular_weekly_alpha,
            'regular_weekly_alpha_annual': regular_weekly_alpha_annual,
            'jensens_weekly_alpha': jensens_weekly_alpha,
            'jensens_weekly_alpha_annual': jensens_weekly_alpha_annual,

            # Risk-adjusted returns
            'weekly_sharpe': weekly_sharpe,
            'weekly_sharpe_annual': weekly_sharpe_annual,
            'weekly_sortino': weekly_sortino,
            'weekly_sortino_annual': weekly_sortino_annual,
            'weekly_information_ratio': weekly_information_ratio,
            'weekly_information_ratio_annual': weekly_information_ratio_annual,

            # Risk metrics
            'weekly_vol_strategy': weekly_vol_strategy,
            'weekly_vol_strategy_annual': weekly_vol_strategy_annual,
            'weekly_vol_market': weekly_vol_market,
            'weekly_vol_market_annual': weekly_vol_market_annual,
            'weekly_max_drawdown': weekly_max_drawdown,
            'weekly_var_95': weekly_var_95,
            'weekly_cvar_95': weekly_cvar_95,

            # Other metrics
            'weekly_correlation': weekly_correlation,
            'weekly_r_squared': weekly_r_squared,
            'weekly_tracking_error': weekly_tracking_error,
            'weekly_tracking_error_annual': weekly_tracking_error * np.sqrt(52),

            # Return metrics
            'weekly_mean_return_strategy': strategy_mean_weekly,
            'weekly_mean_return_market': market_mean_weekly,
            'weekly_total_return_strategy': (1 + strategy_weekly_ret).prod() - 1,
            'weekly_total_return_market': (1 + market_weekly_ret).prod() - 1,

            # Store the series for later use
            'weekly_returns_df': weekly_returns,
            'weekly_prices_df': weekly_prices
        }

        # Display results
        self._display_weekly_metrics()

        return self.results['weekly_metrics']

    def _display_weekly_metrics(self):
        """Display weekly metrics in visually enhanced tables."""
        if 'weekly_metrics' not in self.results:
            return

        weekly = self.results['weekly_metrics']

        # Create main weekly metrics table
        weekly_table = Table(title="📅 Weekly Performance Metrics", box=box.DOUBLE_EDGE)
        weekly_table.add_column("Metric", style="cyan", width=30)
        weekly_table.add_column("Weekly", style="yellow", width=20)
        weekly_table.add_column("Annualized", style="green", width=20)

        # Beta and Alpha
        weekly_table.add_row(
            "Beta (β)",
            f"{weekly['weekly_beta']:.4f}",
            "-"
        )
        weekly_table.add_row(
            "Jensen's Alpha",
            f"{weekly['jensens_weekly_alpha']:.4f}",
            f"{weekly['jensens_weekly_alpha_annual']:.2%}"
        )
        weekly_table.add_row(
            "Regular Alpha",
            f"{weekly['regular_weekly_alpha']:.4f}",
            f"{weekly['regular_weekly_alpha_annual']:.2%}"
        )

        # Risk-adjusted ratios
        weekly_table.add_row(
            "Sharpe Ratio",
            f"{weekly['weekly_sharpe']:.4f}",
            f"{weekly['weekly_sharpe_annual']:.4f}"
        )
        weekly_table.add_row(
            "Sortino Ratio",
            f"{weekly['weekly_sortino']:.4f}",
            f"{weekly['weekly_sortino_annual']:.4f}"
        )
        weekly_table.add_row(
            "Information Ratio",
            f"{weekly['weekly_information_ratio']:.4f}",
            f"{weekly['weekly_information_ratio_annual']:.4f}"
        )

        # Statistical measures
        weekly_table.add_row(
            "R-squared",
            f"{weekly['weekly_r_squared']:.4f}",
            "-"
        )
        weekly_table.add_row(
            "Correlation",
            f"{weekly['weekly_correlation']:.4f}",
            "-"
        )

        console.print(weekly_table)

        # Risk metrics table
        risk_table = Table(title="📊 Weekly Risk Metrics", box=box.SIMPLE_HEAVY)
        risk_table.add_column("Metric", style="cyan", width=30)
        risk_table.add_column("Strategy", style="green", width=20)
        risk_table.add_column("Market", style="yellow", width=20)

        risk_table.add_row(
            "Weekly Volatility",
            f"{weekly['weekly_vol_strategy']:.4f}",
            f"{weekly['weekly_vol_market']:.4f}"
        )
        risk_table.add_row(
            "Annual Volatility",
            f"{weekly['weekly_vol_strategy_annual']:.2%}",
            f"{weekly['weekly_vol_market_annual']:.2%}"
        )
        risk_table.add_row(
            "Total Return",
            f"{weekly['weekly_total_return_strategy']:.2%}",
            f"{weekly['weekly_total_return_market']:.2%}"
        )
        risk_table.add_row(
            "Max Drawdown",
            f"{weekly['weekly_max_drawdown']:.2%}",
            "-"
        )
        risk_table.add_row(
            "VaR 95% (Weekly)",
            f"{weekly['weekly_var_95']:.4%}",
            "-"
        )
        risk_table.add_row(
            "CVaR 95% (Weekly)",
            f"{weekly['weekly_cvar_95']:.4%}",
            "-"
        )

        console.print(risk_table)

        # Summary info
        summary_text = Text()
        summary_text.append(f"✓ Analysis based on {weekly['n_weeks']} weekly observations\n", style="green")
        summary_text.append(f"  Date range: {weekly['date_range']}\n")
        summary_text.append(f"  Weekly tracking error: {weekly['weekly_tracking_error']:.4f} ({weekly['weekly_tracking_error_annual']:.2%} annualized)")

        console.print(Panel(summary_text, title="Weekly Analysis Summary", border_style="blue"))

    def calculate_rolling_beta(self, window=None):
        """
        Calculate rolling beta with progress tracking.
        """
        if window is None:
            window = min(self.rolling_window, len(self.data) // 4)

        console.print(f"\n[bold cyan]Calculating Rolling Beta ({window}-period window)...[/bold cyan]")

        strategy_ret = self.data['strategy_returns']
        market_ret = self.data[f'{ASSET_NAME}_returns']

        rolling_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:

            task = progress.add_task("[cyan]Computing rolling metrics...", total=len(self.data)-window)

            for i in range(window, len(self.data)):
                window_strategy = strategy_ret.iloc[i-window:i]
                window_market = market_ret.iloc[i-window:i]

                valid_data = pd.DataFrame({
                    'strategy': window_strategy,
                    'market': window_market
                }).dropna()

                if len(valid_data) >= 20:
                    # Calculate metrics
                    covariance = np.cov(valid_data['strategy'], valid_data['market'])[0, 1]
                    variance = np.var(valid_data['market'], ddof=1)
                    beta = covariance / variance if variance != 0 else np.nan

                    # Alpha calculation
                    alpha = np.mean(valid_data['strategy']) - beta * np.mean(valid_data['market'])

                    # Correlation
                    correlation = np.corrcoef(valid_data['strategy'], valid_data['market'])[0, 1]
                    r_squared = correlation ** 2

                    rolling_results.append({
                        'date': self.data['date'].iloc[i],
                        'rolling_beta': beta,
                        'rolling_alpha': alpha,
                        'rolling_r_squared': r_squared,
                        'asset_price': self.data['close'].iloc[i],
                        'strategy_value': self.data['results'].iloc[i]
                    })

                progress.update(task, advance=1)

        rolling_df = pd.DataFrame(rolling_results)
        self.results['rolling_analysis'] = rolling_df

        # Display summary
        if len(rolling_df) > 0:
            summary_text = Text()
            summary_text.append(f"✓ Calculated {len(rolling_df):,} rolling observations\n", style="green")
            summary_text.append(f"  Beta range: [{rolling_df['rolling_beta'].min():.4f}, {rolling_df['rolling_beta'].max():.4f}]\n")
            summary_text.append(f"  Mean beta: {rolling_df['rolling_beta'].mean():.4f}\n")
            summary_text.append(f"  Beta volatility: {rolling_df['rolling_beta'].std():.4f}")

            console.print(Panel(summary_text, title="Rolling Analysis Summary", border_style="green"))

        return rolling_df

    def detect_market_regimes(self):
        """
        Detect market regimes based on volatility and calculate regime-specific betas.
        """
        console.print("\n[bold cyan]Detecting Market Regimes...[/bold cyan]")

        market_ret = self.data[f'{ASSET_NAME}_returns'].dropna()

        # Calculate rolling volatility (21-day window)
        vol_window = int(21 * self.periods_per_day)  # 21 days worth of periods
        rolling_vol = market_ret.rolling(window=vol_window).std()

        # Define regime thresholds
        vol_high = rolling_vol.quantile(0.75)
        vol_low = rolling_vol.quantile(0.25)

        # Classify regimes
        regimes = []
        for vol in rolling_vol:
            if pd.isna(vol):
                regimes.append('Unknown')
            elif vol > vol_high:
                regimes.append('High Volatility')
            elif vol < vol_low:
                regimes.append('Low Volatility')
            else:
                regimes.append('Medium Volatility')

        regime_df = pd.DataFrame({
            'date': self.data['date'],
            'volatility_regime': regimes,
            'rolling_volatility': rolling_vol
        })

        # Calculate beta for each regime
        regime_betas = {}
        regime_table = Table(title="📊 Regime-Specific Betas", box=box.ROUNDED)
        regime_table.add_column("Market Regime", style="cyan")
        regime_table.add_column("Observations", style="yellow")
        regime_table.add_column("Beta", style="green")
        regime_table.add_column("% of Time", style="white")

        for regime in ['High Volatility', 'Medium Volatility', 'Low Volatility']:
            regime_mask = regime_df['volatility_regime'] == regime
            if regime_mask.sum() > 20:
                regime_data = self.data.loc[regime_mask]

                strategy_ret = regime_data['strategy_returns'].dropna()
                market_ret = regime_data[f'{ASSET_NAME}_returns'].dropna()

                if len(strategy_ret) > 10 and len(market_ret) > 10:
                    aligned = pd.DataFrame({'strategy': strategy_ret, 'market': market_ret}).dropna()

                    if len(aligned) > 10:
                        covariance = np.cov(aligned['strategy'], aligned['market'])[0, 1]
                        variance = np.var(aligned['market'], ddof=1)
                        beta = covariance / variance if variance != 0 else np.nan
                        regime_betas[regime] = beta

                        regime_table.add_row(
                            regime,
                            f"{regime_mask.sum():,}",
                            f"{beta:.4f}",
                            f"{(regime_mask.sum() / len(regime_df)) * 100:.1f}%"
                        )

        console.print(regime_table)

        self.results['regime_analysis'] = {
            'regime_data': regime_df,
            'regime_betas': regime_betas
        }

        return regime_df, regime_betas

    def create_visualizations(self):
        """
        Create comprehensive visualizations and save to results/html folder.
        """
        console.print("\n[bold cyan]Creating Visualizations...[/bold cyan]")

        # Main dashboard
        self._create_main_dashboard()

        # Rolling beta detailed plot
        self._create_rolling_beta_plot()

        # Price plot with beta shading
        self._create_price_beta_plot()

        console.print("[green]✓ All visualizations created successfully![/green]")

    def _create_main_dashboard(self):
        """Create the main dashboard with 8 panels including comprehensive metrics table."""
        fig = make_subplots(
            rows=5, cols=2,
            subplot_titles=[
                '<b>Beta Regression Analysis</b><br><sub>Strategy vs Market Returns</sub>',
                f'<b>{ASSET_NAME} Price with Beta Regimes</b><br><sub>Low (Red) | Medium (Orange) | High (Green)</sub>',
                '<b>Return Distribution Comparison</b><br><sub>Strategy (Blue) vs Market (Orange)</sub>',
                '<b>Risk-Return Profile</b><br><sub>Annualized Metrics</sub>',
                '<b>Drawdown Analysis</b><br><sub>Peak-to-Trough Declines</sub>',
                '<b>Volatility Regime Analysis</b><br><sub>Market Conditions Over Time</sub>',
                '<b>Weekly vs Daily Returns</b><br><sub>Time Aggregation Comparison</sub>',
                '<b>Weekly Performance Scatter</b><br><sub>Strategy vs Market Weekly Returns</sub>',
                '<b>Comprehensive Metrics Summary</b><br><sub>All Key Performance Indicators</sub>'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"type": "table", "colspan": 2}, None]  # Metrics table spans both columns
            ],
            vertical_spacing=0.06,
            horizontal_spacing=0.08,
            row_heights=[0.20, 0.20, 0.20, 0.20, 0.20]
        )

        # Get data
        strategy_ret = self.data['strategy_returns'].dropna()
        market_ret = self.data[f'{ASSET_NAME}_returns'].dropna()
        aligned_data = pd.DataFrame({'strategy': strategy_ret, 'market': market_ret}).dropna()

        # 1. Beta Regression
        beta_stats = self.results['beta_analysis']
        x_range = np.linspace(aligned_data['market'].min(), aligned_data['market'].max(), 1000)
        y_line = beta_stats['alpha_ols'] + beta_stats['beta_ols'] * x_range

        fig.add_trace(
            go.Scatter(
                x=aligned_data['market'],
                y=aligned_data['strategy'],
                mode='markers',
                marker=dict(size=4, opacity=0.6, color='blue'),
                name='Data Points',
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
                name=f'β = {beta_stats["beta_ols"]:.4f}',
                showlegend=False
            ),
            row=1, col=1
        )

        # 2. XRP Price with Beta Shading
        if 'rolling_analysis' in self.results:
            rolling_data = self.results['rolling_analysis']

            # Add price line
            fig.add_trace(
                go.Scatter(
                    x=rolling_data['date'],
                    y=rolling_data['asset_price'],
                    mode='lines',
                    line=dict(color='black', width=2),
                    name=f'{ASSET_NAME} Price',
                    showlegend=False
                ),
                row=1, col=2
            )

            # Add beta level annotations
            beta_25th = rolling_data['rolling_beta'].quantile(0.25)
            beta_75th = rolling_data['rolling_beta'].quantile(0.75)

            # Create beta level colors for background
            rolling_data_copy = rolling_data.copy()
            rolling_data_copy['beta_level'] = 'Medium'
            rolling_data_copy.loc[rolling_data_copy['rolling_beta'] < beta_25th, 'beta_level'] = 'Low'
            rolling_data_copy.loc[rolling_data_copy['rolling_beta'] > beta_75th, 'beta_level'] = 'High'

            # Add colored scatter points to represent beta levels
            for level, color in [('Low', 'red'), ('Medium', 'orange'), ('High', 'green')]:
                level_data = rolling_data_copy[rolling_data_copy['beta_level'] == level]
                if len(level_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=level_data['date'],
                            y=level_data['asset_price'],
                            mode='markers',
                            marker=dict(size=2, color=color, opacity=0.3),
                            name=f'{level} Beta',
                            showlegend=False
                        ),
                        row=1, col=2
                    )

        # 3. Return Distribution Comparison
        fig.add_trace(
            go.Histogram(
                x=strategy_ret,
                name='Strategy Returns',
                opacity=0.7,
                nbinsx=50,
                showlegend=False,
                marker_color='blue'
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Histogram(
                x=market_ret,
                name=f'{ASSET_NAME} Returns',
                opacity=0.7,
                nbinsx=50,
                showlegend=False,
                marker_color='orange'
            ),
            row=2, col=1
        )

        # 4. Risk-Return Profile
        annual_return_strategy = np.mean(strategy_ret) * self.periods_per_year
        annual_vol_strategy = np.std(strategy_ret, ddof=1) * self.annualization_factor
        annual_return_market = np.mean(market_ret) * self.periods_per_year
        annual_vol_market = np.std(market_ret, ddof=1) * self.annualization_factor

        fig.add_trace(
            go.Scatter(
                x=[annual_vol_strategy, annual_vol_market],
                y=[annual_return_strategy, annual_return_market],
                mode='markers+text',
                marker=dict(size=15),
                text=['Strategy', ASSET_NAME],
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
                name='Drawdown %',
                showlegend=False
            ),
            row=3, col=1
        )

        # 6. Volatility Regime Analysis
        if 'regime_analysis' in self.results:
            regime_data = self.results['regime_analysis']['regime_data']
            volatility_colors = {'High Volatility': 'red', 'Medium Volatility': 'orange', 'Low Volatility': 'green'}

            for regime, color in volatility_colors.items():
                regime_mask = regime_data['volatility_regime'] == regime
                if regime_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=regime_data.loc[regime_mask, 'date'],
                            y=regime_data.loc[regime_mask, 'rolling_volatility'] * self.annualization_factor,
                            mode='markers',
                            marker=dict(color=color, size=4),
                            name=regime,
                            showlegend=False
                        ),
                        row=3, col=2
                    )

        # 7. Weekly vs Daily Returns Comparison
        if 'weekly_metrics' in self.results:
            weekly_df = self.results['weekly_metrics']['weekly_returns_df']

            # Create cumulative returns for both daily and weekly
            daily_cumret = (1 + self.data['strategy_returns']).cumprod()
            weekly_cumret = (1 + weekly_df['strategy_weekly_returns']).cumprod()

            # Normalize to start at 1
            daily_cumret = daily_cumret / daily_cumret.iloc[0]
            weekly_cumret = weekly_cumret / weekly_cumret.iloc[0]

            # Plot daily cumulative returns
            fig.add_trace(
                go.Scatter(
                    x=self.data['date'],
                    y=daily_cumret,
                    mode='lines',
                    line=dict(color='blue', width=1),
                    name='Daily Returns',
                    showlegend=False
                ),
                row=4, col=1
            )

            # Plot weekly cumulative returns
            fig.add_trace(
                go.Scatter(
                    x=weekly_df.index,
                    y=weekly_cumret,
                    mode='lines+markers',
                    line=dict(color='red', width=2),
                    marker=dict(size=4),
                    name='Weekly Returns',
                    showlegend=False
                ),
                row=4, col=1
            )

        # 8. Weekly Performance Scatter
        if 'weekly_metrics' in self.results:
            weekly_df = self.results['weekly_metrics']['weekly_returns_df']
            weekly_beta = self.results['weekly_metrics']['weekly_beta']
            weekly_alpha = self.results['weekly_metrics']['jensens_weekly_alpha']

            # Weekly scatter plot
            fig.add_trace(
                go.Scatter(
                    x=weekly_df[f'{ASSET_NAME}_weekly_returns'],
                    y=weekly_df['strategy_weekly_returns'],
                    mode='markers',
                    marker=dict(size=8, opacity=0.7, color='green'),
                    name='Weekly Returns',
                    showlegend=False
                ),
                row=4, col=2
            )

            # Add weekly regression line
            x_range_weekly = np.linspace(
                weekly_df[f'{ASSET_NAME}_weekly_returns'].min(),
                weekly_df[f'{ASSET_NAME}_weekly_returns'].max(),
                100
            )
            y_line_weekly = weekly_alpha + weekly_beta * x_range_weekly

            fig.add_trace(
                go.Scatter(
                    x=x_range_weekly,
                    y=y_line_weekly,
                    mode='lines',
                    line=dict(color='darkgreen', width=2, dash='dash'),
                    name=f'Weekly β = {weekly_beta:.4f}',
                    showlegend=False
                ),
                row=4, col=2
            )

        # Update axes labels
        fig.update_xaxes(title_text=f"{ASSET_NAME} Returns", row=1, col=1)
        fig.update_yaxes(title_text="Strategy Returns", row=1, col=1)

        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text=f"{ASSET_NAME} Price ($)", row=1, col=2)

        fig.update_xaxes(title_text="Returns", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)

        fig.update_xaxes(title_text="Annual Volatility", row=2, col=2)
        fig.update_yaxes(title_text="Annual Return", row=2, col=2)

        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=3, col=1)

        fig.update_xaxes(title_text="Date", row=3, col=2)
        fig.update_yaxes(title_text="Annual Volatility", row=3, col=2)

        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Cumulative Return", row=4, col=1)

        fig.update_xaxes(title_text=f"{ASSET_NAME} Weekly Returns", row=4, col=2)
        fig.update_yaxes(title_text="Strategy Weekly Returns", row=4, col=2)

        # 9. Comprehensive Metrics Table
        # Prepare all metrics data
        metrics_data = []

        # Data Summary
        time_span = self.data['date'].max() - self.data['date'].min()
        metrics_data.append(["<b>DATA SUMMARY</b>", "", "", ""])
        metrics_data.append(["Date Range", f"{self.data['date'].min().strftime('%Y-%m-%d')} to {self.data['date'].max().strftime('%Y-%m-%d')}", "", ""])
        metrics_data.append(["Total Periods", f"{len(self.data):,}", "Time Span", f"{time_span.days} days"])
        metrics_data.append(["Data Frequency", self.data_frequency, "Periods/Year", f"{self.periods_per_year:,}"])

        # Core Beta & Alpha Metrics
        if 'beta_analysis' in self.results:
            beta_stats = self.results['beta_analysis']
            metrics_data.append(["", "", "", ""])  # Spacer
            metrics_data.append(["<b>CORE BETA & ALPHA METRICS</b>", "", "", ""])
            metrics_data.append(["Beta (β)", f"{beta_stats['beta_ols']:.4f}", "95% CI", f"[{beta_stats['beta_ci_lower']:.4f}, {beta_stats['beta_ci_upper']:.4f}]"])
            metrics_data.append(["Jensen's Alpha (Annual)", f"{beta_stats['jensens_alpha_annual']:.2%}", "Regular Alpha (Annual)", f"{beta_stats['regular_alpha_annual']:.2%}"])
            metrics_data.append(["R-squared", f"{beta_stats['r_squared']:.4f}", "Correlation", f"{beta_stats['correlation']:.4f}"])

        # Risk-Adjusted Performance Ratios
        if 'performance_ratios' in self.results:
            ratios = self.results['performance_ratios']
            metrics_data.append(["", "", "", ""])  # Spacer
            metrics_data.append(["<b>RISK-ADJUSTED RATIOS (Rf=0)</b>", "", "", ""])
            metrics_data.append(["Sharpe Ratio", f"{ratios['sharpe_ratio_annual']:.4f}", "Sortino Ratio", f"{ratios['sortino_ratio_annual']:.4f}"])
            metrics_data.append(["Treynor Ratio", f"{ratios['treynor_ratio_annual']:.2%}", "Information Ratio", f"{ratios['information_ratio_annual']:.4f}"])
            metrics_data.append(["Calmar Ratio", f"{ratios['calmar_ratio']:.4f}", "Tracking Error", f"{ratios['tracking_error_annual']:.2%}"])

        # Risk Metrics
        if 'performance_ratios' in self.results:
            ratios = self.results['performance_ratios']
            metrics_data.append(["", "", "", ""])  # Spacer
            metrics_data.append(["<b>RISK METRICS</b>", "", "", ""])
            metrics_data.append(["Max DD (Strategy)", f"{ratios['max_drawdown_strategy']:.2%}", "Max DD (Asset)", f"{ratios['max_drawdown_asset']:.2%}"])
            metrics_data.append(["Initial DD (Strategy)", f"{ratios['initial_drawdown_strategy']:.2%}", "Initial DD (Asset)", f"{ratios['initial_drawdown_asset']:.2%}"])
            metrics_data.append(["VaR 95%", f"{ratios['var_95']:.4%}", "CVaR 95%", f"{ratios['cvar_95']:.4%}"])
            metrics_data.append(["Skewness", f"{ratios['skewness']:.4f}", "Kurtosis", f"{ratios['kurtosis']:.4f}"])

        # Performance Summary
        if 'performance_ratios' in self.results:
            ratios = self.results['performance_ratios']
            metrics_data.append(["", "", "", ""])  # Spacer
            metrics_data.append(["<b>PERFORMANCE SUMMARY</b>", "", "", ""])
            metrics_data.append(["Strategy Total Return", f"{ratios['strategy_gain']:.2%}", "Market Total Return", f"{ratios['market_gain']:.2%}"])
            metrics_data.append(["Strategy Down %", f"{ratios['down_percentage_strategy']:.1f}%", "Market Down %", f"{ratios['down_percentage_asset']:.1f}%"])

        # Weekly Metrics
        if 'weekly_metrics' in self.results:
            weekly = self.results['weekly_metrics']
            metrics_data.append(["", "", "", ""])  # Spacer
            metrics_data.append(["<b>WEEKLY METRICS COMPARISON</b>", "", "", ""])
            metrics_data.append(["Weekly Beta", f"{weekly['weekly_beta']:.4f}", "Daily Beta", f"{beta_stats['beta_ols']:.4f}"])
            metrics_data.append(["Weekly Sharpe (Annual)", f"{weekly['weekly_sharpe_annual']:.4f}", "Daily Sharpe (Annual)", f"{ratios['sharpe_ratio_annual']:.4f}"])
            metrics_data.append(["Weekly Sortino (Annual)", f"{weekly['weekly_sortino_annual']:.4f}", "Daily Sortino (Annual)", f"{ratios['sortino_ratio_annual']:.4f}"])
            metrics_data.append(["Weekly Jensen's α (Annual)", f"{weekly['jensens_weekly_alpha_annual']:.2%}", "Daily Jensen's α (Annual)", f"{beta_stats['jensens_alpha_annual']:.2%}"])
            metrics_data.append(["Weekly R-squared", f"{weekly['weekly_r_squared']:.4f}", "Daily R-squared", f"{beta_stats['r_squared']:.4f}"])
            metrics_data.append(["Weekly Vol (Annual)", f"{weekly['weekly_vol_strategy_annual']:.2%}", "Daily Vol (Annual)", f"{np.std(self.data['strategy_returns'], ddof=1) * self.annualization_factor:.2%}"])
            metrics_data.append(["Weekly Observations", f"{weekly['n_weeks']}", "Daily Observations", f"{len(self.data)}"])

        # Regime-Specific Betas
        if 'regime_analysis' in self.results:
            regime_betas = self.results['regime_analysis']['regime_betas']
            metrics_data.append(["", "", "", ""])  # Spacer
            metrics_data.append(["<b>REGIME-SPECIFIC BETAS</b>", "", "", ""])
            for regime, beta in regime_betas.items():
                metrics_data.append([regime, f"{beta:.4f}", "", ""])

        # Create table - transpose the data properly
        col1, col2, col3, col4 = [], [], [], []
        for row in metrics_data:
            col1.append(row[0])
            col2.append(row[1])
            col3.append(row[2])
            col4.append(row[3])

        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Value</b>', '<b>Metric</b>', '<b>Value</b>'],
                    fill_color='darkblue',
                    align=['left', 'center', 'left', 'center'],
                    font=dict(size=12, color='white'),
                    height=25
                ),
                cells=dict(
                    values=[col1, col2, col3, col4],
                    fill_color=[
                        ['lightblue' if '<b>' in str(val) else 'white' for val in col1],
                        ['lightblue' if col1[i] and '<b>' in str(col1[i]) else 'white' for i, val in enumerate(col2)],
                        ['lightblue' if '<b>' in str(val) else 'white' for val in col3],
                        ['lightblue' if col3[i] and '<b>' in str(col3[i]) else 'white' for i, val in enumerate(col4)]
                    ],
                    align=['left', 'center', 'left', 'center'],
                    font=dict(size=11),
                    height=22
                )
            ),
            row=5, col=1
        )

        # Update layout
        fig.update_layout(
            height=2000,  # Increased height for table and new plots
            width=1600,
            title=dict(
                text='<b>Enhanced Beta Analysis Dashboard</b><br><sub>Comprehensive Performance & Risk Metrics</sub>',
                x=0.5,
                font=dict(size=26, color='darkblue', family="Arial Black, sans-serif")
            ),
            showlegend=False,
            template='plotly_white',
            font=dict(family="Arial, sans-serif"),
            paper_bgcolor='rgba(240,240,240,0.9)',
            plot_bgcolor='white'
        )

        # Add annotations for better visual guidance
        fig.add_annotation(
            text=f"Analysis Period: {self.data['date'].min().strftime('%Y-%m-%d')} to {self.data['date'].max().strftime('%Y-%m-%d')} | Data Frequency: {self.data_frequency}",
            xref="paper", yref="paper",
            x=0.5, y=0.96,
            showarrow=False,
            font=dict(size=12, color="gray")
        )

        # Save
        output_path = self.html_dir / 'beta_analysis_dashboard_v3.html'
        fig.write_html(str(output_path))
        self.logger.info(f"Main dashboard saved to: {output_path}")

    def _create_rolling_beta_plot(self):
        """Create detailed rolling beta analysis plot."""
        if 'rolling_analysis' not in self.results:
            return

        rolling_data = self.results['rolling_analysis']

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add rolling beta
        fig.add_trace(
            go.Scatter(
                x=rolling_data['date'],
                y=rolling_data['rolling_beta'],
                mode='lines',
                line=dict(color='blue', width=2),
                name='Rolling Beta'
            ),
            secondary_y=False
        )

        # Add asset price
        fig.add_trace(
            go.Scatter(
                x=rolling_data['date'],
                y=rolling_data['asset_price'],
                mode='lines',
                line=dict(color='orange', width=1),
                name=f'{ASSET_NAME} Price'
            ),
            secondary_y=True
        )

        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Rolling Beta", secondary_y=False)
        fig.update_yaxes(title_text=f"{ASSET_NAME} Price ($)", secondary_y=True)

        fig.update_layout(
            title=f"Rolling Beta Analysis with {ASSET_NAME} Price",
            template='plotly_white',
            height=600
        )

        output_path = self.html_dir / 'rolling_beta_detailed_v3.html'
        fig.write_html(str(output_path))
        self.logger.info(f"Rolling beta plot saved to: {output_path}")

    def _create_price_beta_plot(self):
        """Create price plot with beta-based background shading."""
        if 'rolling_analysis' not in self.results:
            return

        rolling_data = self.results['rolling_analysis']

        # Classify beta levels
        beta_25th = rolling_data['rolling_beta'].quantile(0.25)
        beta_75th = rolling_data['rolling_beta'].quantile(0.75)

        fig = go.Figure()

        # Add background shading based on beta levels
        rolling_data = rolling_data.copy()
        rolling_data['beta_level'] = 'Medium'
        rolling_data.loc[rolling_data['rolling_beta'] < beta_25th, 'beta_level'] = 'Low'
        rolling_data.loc[rolling_data['rolling_beta'] > beta_75th, 'beta_level'] = 'High'

        # Create shading
        current_level = None
        start_date = None

        for i, row in rolling_data.iterrows():
            if current_level != row['beta_level']:
                if current_level is not None and start_date is not None:
                    color = {
                        'High': 'rgba(0, 255, 0, 0.2)',
                        'Medium': 'rgba(255, 165, 0, 0.2)',
                        'Low': 'rgba(255, 0, 0, 0.2)'
                    }[current_level]

                    fig.add_vrect(
                        x0=start_date,
                        x1=row['date'],
                        fillcolor=color,
                        layer="below",
                        line_width=0,
                    )

                current_level = row['beta_level']
                start_date = row['date']

        # Add final segment
        if current_level is not None and start_date is not None:
            color = {
                'High': 'rgba(0, 255, 0, 0.2)',
                'Medium': 'rgba(255, 165, 0, 0.2)',
                'Low': 'rgba(255, 0, 0, 0.2)'
            }[current_level]

            fig.add_vrect(
                x0=start_date,
                x1=rolling_data['date'].iloc[-1],
                fillcolor=color,
                layer="below",
                line_width=0,
            )

        # Add price line
        fig.add_trace(
            go.Scatter(
                x=rolling_data['date'],
                y=rolling_data['asset_price'],
                mode='lines',
                line=dict(color='black', width=2),
                name=f'{ASSET_NAME} Price'
            )
        )

        fig.update_layout(
            title=f'{ASSET_NAME} Price with Beta-Based Background Shading',
            xaxis_title='Date',
            yaxis_title=f'{ASSET_NAME} Price ($)',
            template='plotly_white',
            height=600
        )

        output_path = self.html_dir / f'{ASSET_NAME.lower()}_price_beta_shaded_v3.html'
        fig.write_html(str(output_path))
        self.logger.info(f"Price beta plot saved to: {output_path}")

    def save_results(self):
        """
        Save all results to CSV files in organized structure.
        """
        console.print("\n[bold cyan]Saving Results...[/bold cyan]")

        # Comprehensive results
        comprehensive_results = {}

        # Add beta analysis
        if 'beta_analysis' in self.results:
            comprehensive_results.update(self.results['beta_analysis'])

        # Add performance ratios
        if 'performance_ratios' in self.results:
            comprehensive_results.update(self.results['performance_ratios'])

        # Add regime betas
        if 'regime_analysis' in self.results:
            regime_betas = self.results['regime_analysis']['regime_betas']
            for regime, beta in regime_betas.items():
                regime_key = regime.lower().replace(' ', '_') + '_beta'
                comprehensive_results[regime_key] = beta

        # Add weekly metrics
        if 'weekly_metrics' in self.results:
            weekly = self.results['weekly_metrics']
            # Add key weekly metrics with 'weekly_' prefix to distinguish from daily
            weekly_metrics_to_save = {
                'weekly_beta': weekly['weekly_beta'],
                'weekly_alpha_jensen_annual': weekly['jensens_weekly_alpha_annual'],
                'weekly_alpha_regular_annual': weekly['regular_weekly_alpha_annual'],
                'weekly_sharpe_annual': weekly['weekly_sharpe_annual'],
                'weekly_sortino_annual': weekly['weekly_sortino_annual'],
                'weekly_information_ratio_annual': weekly['weekly_information_ratio_annual'],
                'weekly_r_squared': weekly['weekly_r_squared'],
                'weekly_correlation': weekly['weekly_correlation'],
                'weekly_volatility_annual': weekly['weekly_vol_strategy_annual'],
                'weekly_max_drawdown': weekly['weekly_max_drawdown'],
                'weekly_var_95': weekly['weekly_var_95'],
                'weekly_cvar_95': weekly['weekly_cvar_95'],
                'weekly_total_return': weekly['weekly_total_return_strategy'],
                'weekly_n_observations': weekly['n_weeks']
            }
            comprehensive_results.update(weekly_metrics_to_save)

        # Save comprehensive results
        results_df = pd.DataFrame([comprehensive_results])
        results_path = self.csv_dir / 'beta_analysis_results_v3.csv'
        results_df.to_csv(results_path, index=False)

        # Save rolling analysis
        if 'rolling_analysis' in self.results:
            rolling_path = self.csv_dir / 'rolling_beta_analysis_v3.csv'
            self.results['rolling_analysis'].to_csv(rolling_path, index=False)

        # Display save summary
        save_table = Table(title="📁 Saved Files", box=box.SIMPLE)
        save_table.add_column("Type", style="cyan")
        save_table.add_column("Location", style="green")

        # Resolve paths to be absolute before making them relative
        cwd = Path.cwd()
        save_table.add_row("Results CSV", str(results_path.resolve().relative_to(cwd)))
        save_table.add_row("Rolling CSV", str(rolling_path.resolve().relative_to(cwd)))

        # Also resolve html paths for consistency
        dashboard_path = self.html_dir / 'beta_analysis_dashboard_v3.html'
        rolling_plot_path = self.html_dir / 'rolling_beta_detailed_v3.html'
        price_plot_path = self.html_dir / 'btc_price_beta_shaded_v3.html'

        save_table.add_row("Main Dashboard", str(dashboard_path.resolve().relative_to(cwd)))
        save_table.add_row("Rolling Beta Plot", str(rolling_plot_path.resolve().relative_to(cwd)))
        save_table.add_row("Price Beta Plot", str(price_plot_path.resolve().relative_to(cwd)))

        console.print(save_table)

        self.logger.info("All results saved successfully")

    def print_final_summary(self):
        """
        Print a comprehensive final summary with interpretations.
        """
        console.print("\n" + "="*80)
        console.print("[bold cyan]FINAL ANALYSIS SUMMARY[/bold cyan]")
        console.print("="*80 + "\n")

        # Strategy characterization
        beta = self.results['beta_analysis']['beta_ols']
        r_squared = self.results['beta_analysis']['r_squared']

        char_text = Text()

        if beta < 0.5:
            char_text.append("🛡️  LOW BETA DEFENSIVE STRATEGY\n", style="bold yellow")
            char_text.append(f"Beta of {beta:.3f} indicates minimal market sensitivity\n")
        elif beta < 0.8:
            char_text.append("⚖️  MODERATE BETA BALANCED STRATEGY\n", style="bold blue")
            char_text.append(f"Beta of {beta:.3f} shows moderate market correlation\n")
        elif beta < 1.2:
            char_text.append("📊 MARKET-NEUTRAL STRATEGY\n", style="bold green")
            char_text.append(f"Beta of {beta:.3f} closely tracks market movements\n")
        else:
            char_text.append("🚀 HIGH BETA AGGRESSIVE STRATEGY\n", style="bold red")
            char_text.append(f"Beta of {beta:.3f} amplifies market movements\n")

        # R-squared interpretation
        if r_squared > 0.7:
            char_text.append(f"R² of {r_squared:.1%} indicates strong market correlation", style="green")
        elif r_squared > 0.3:
            char_text.append(f"R² of {r_squared:.1%} shows moderate market correlation", style="yellow")
        else:
            char_text.append(f"R² of {r_squared:.1%} suggests weak market correlation", style="red")

        console.print(Panel(char_text, title="Strategy Characterization", border_style="cyan"))

        # Key insights
        insights = []

        # Alpha insights
        jensen_alpha = self.results['beta_analysis']['jensens_alpha_annual']
        if jensen_alpha > 0.05:
            insights.append(("✅", "Positive alpha generation", "green"))
        elif jensen_alpha < -0.05:
            insights.append(("❌", "Negative alpha (underperformance)", "red"))
        else:
            insights.append(("➖", "Negligible alpha generation", "yellow"))

        # Risk-adjusted performance
        sharpe = self.results['performance_ratios']['sharpe_ratio_annual']
        if sharpe < 0.5:
            insights.append(("⚠️", "Poor risk-adjusted returns", "red"))
        elif sharpe < 1:
            insights.append(("📊", "Moderate risk-adjusted returns", "yellow"))
        else:
            insights.append(("🎯", "Good risk-adjusted returns", "green"))

        # Drawdown
        max_dd = self.results['performance_ratios']['max_drawdown_strategy']
        if max_dd < -0.20:
            insights.append(("📉", f"Significant drawdown risk ({max_dd:.1%})", "red"))
        else:
            insights.append(("📈", f"Moderate drawdown risk ({max_dd:.1%})", "yellow"))

        insights_table = Table(title="🔍 Key Insights", box=box.MINIMAL)
        insights_table.add_column("", width=3)
        insights_table.add_column("Insight", style="white")

        for icon, text, color in insights:
            insights_table.add_row(icon, f"[{color}]{text}[/{color}]")

        console.print(insights_table)

        # Performance Comparison
        comparison_table = Table(title="📊 Strategy vs Market Comparison", box=box.ROUNDED)
        comparison_table.add_column("Metric", style="cyan")
        comparison_table.add_column("Strategy", style="green")
        comparison_table.add_column("Market", style="yellow")

        strategy_gain = self.results['performance_ratios']['strategy_gain']
        market_gain = self.results['performance_ratios']['market_gain']
        comparison_table.add_row("Total Return", f"{strategy_gain:.2%}", f"{market_gain:.2%}")

        max_dd_strategy = self.results['performance_ratios']['max_drawdown_strategy']
        max_dd_asset = self.results['performance_ratios']['max_drawdown_asset']
        comparison_table.add_row("Maximum Drawdown", f"{max_dd_strategy:.2%}", f"{max_dd_asset:.2%}")

        init_dd_strategy = self.results['performance_ratios']['initial_drawdown_strategy']
        init_dd_asset = self.results['performance_ratios']['initial_drawdown_asset']
        comparison_table.add_row("Initial Drawdown", f"{init_dd_strategy:.2%}", f"{init_dd_asset:.2%}")

        down_pct_strategy = self.results['performance_ratios']['down_percentage_strategy']
        down_pct_asset = self.results['performance_ratios']['down_percentage_asset']
        comparison_table.add_row("Time Below Initial", f"{down_pct_strategy:.1f}%", f"{down_pct_asset:.1f}%")

        console.print(comparison_table)

        # Weekly vs Daily Metrics Comparison (if available)
        if 'weekly_metrics' in self.results:
            weekly = self.results['weekly_metrics']

            weekly_comparison = Table(title="📊 Weekly vs Daily Metrics", box=box.MINIMAL)
            weekly_comparison.add_column("Metric", style="cyan")
            weekly_comparison.add_column("Weekly", style="green")
            weekly_comparison.add_column("Daily", style="yellow")
            weekly_comparison.add_column("Insight", style="white")

            # Beta comparison
            beta_diff = abs(weekly['weekly_beta'] - beta)
            beta_insight = "Consistent" if beta_diff < 0.1 else "Different risk profiles"
            weekly_comparison.add_row(
                "Beta",
                f"{weekly['weekly_beta']:.4f}",
                f"{beta:.4f}",
                beta_insight
            )

            # Sharpe comparison
            weekly_sharpe = weekly['weekly_sharpe_annual']
            daily_sharpe = self.results['performance_ratios']['sharpe_ratio_annual']
            sharpe_insight = "Weekly smoother" if weekly_sharpe > daily_sharpe else "Daily captures more"
            weekly_comparison.add_row(
                "Sharpe (Annual)",
                f"{weekly_sharpe:.4f}",
                f"{daily_sharpe:.4f}",
                sharpe_insight
            )

            # R-squared comparison
            r2_insight = "Similar correlation" if abs(weekly['weekly_r_squared'] - r_squared) < 0.1 else "Different patterns"
            weekly_comparison.add_row(
                "R-squared",
                f"{weekly['weekly_r_squared']:.4f}",
                f"{r_squared:.4f}",
                r2_insight
            )

            console.print(weekly_comparison)

        # Action recommendations
        console.print("\n[bold]📋 Recommendations:[/bold]")

        if beta < 0.3 and r_squared < 0.3:
            console.print("  • Investigate potential data issues or strategy independence")

        if sharpe < 0.5:
            console.print("  • Consider risk reduction or return enhancement strategies")

        if jensen_alpha < 0:
            console.print("  • Review strategy implementation and costs")

        if abs(self.results['performance_ratios']['skewness']) > 1:
            console.print("  • Address return distribution asymmetry")

        # Weekly-specific recommendations
        if 'weekly_metrics' in self.results:
            weekly = self.results['weekly_metrics']
            if abs(weekly['weekly_beta'] - beta) > 0.2:
                console.print("  • Large beta difference between weekly/daily - consider time horizon")

            if weekly['weekly_sharpe_annual'] > daily_sharpe * 1.2:
                console.print("  • Weekly metrics show better risk-adjusted returns - consider longer holding periods")

        console.print("\n[green]✅ Analysis complete! Check the results/ folder for all outputs.[/green]")


def main(file_path, rolling_window=18144, confidence_level=0.95, data_frequency='5min', multi_window_analysis=False):
    """
    Main execution function for enhanced beta analysis.

    Args:
        file_path (str): Path to the data file
        rolling_window (int): Window size for rolling calculations (default: 18144 = 63 days for 5min data)
        confidence_level (float): Confidence level for statistical tests
        data_frequency (str): Data frequency ('5min', 'hourly', 'daily')
        multi_window_analysis (bool): Whether to perform multi-window analysis
    """
    try:
        # Header
        console.print("\n[bold magenta]🚀 ENHANCED BETA ANALYZER v3.0[/bold magenta]")
        console.print("[dim]Advanced Performance Metrics with Visual Enhancements[/dim]\n")

        # Initialize analyzer
        analyzer = EnhancedBetaAnalyzer(
            confidence_level=confidence_level,
            rolling_window=rolling_window,
            risk_free_rate=0.0,  # As requested
            data_frequency=data_frequency
        )

        # Load and validate data
        analyzer.load_and_validate_data(file_path, validate_stationarity=True)

        # Calculate comprehensive beta and alpha metrics
        analyzer.calculate_comprehensive_beta()

        # Calculate performance ratios
        analyzer.calculate_performance_ratios()

        # Calculate weekly metrics
        analyzer.calculate_weekly_metrics()

        # Calculate rolling beta
        analyzer.calculate_rolling_beta()

        # Detect market regimes
        analyzer.detect_market_regimes()

        # Create visualizations
        analyzer.create_visualizations()

        # Save all results
        analyzer.save_results()

        # Print final summary
        analyzer.print_final_summary()

    except Exception as e:
        console.print(f"[bold red]❌ Error occurred: {str(e)}[/bold red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # File path
    file_path = '/Users/samnazari/Downloads/emails/portfolio_summary_updated.csv'

    # Run analysis
    main(
        file_path=file_path,
        rolling_window=6048,     #18144,  # 63 days for 5-minute data (63 * 288 periods/day)
        confidence_level=0.95,
        data_frequency='15min',
        multi_window_analysis=False
    )
