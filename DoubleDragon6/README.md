# DoubleDragon6 Trading System

DoubleDragon6 is a grid-based trading system that implements a dual-agent approach (HighKick and AxeKick) for cryptocurrency and stock markets.

## Configuration System

The system now uses a YAML-based configuration system that makes it easy to change parameters without modifying the code.

### Getting Started

1. Install required packages:
   ```
   pip install pyyaml pandas numpy plotly scipy
   ```

2. Edit the `config.yaml` file to customize your trading parameters

3. Run the system using the configuration:
   ```
   python Example_with_Config.py
   ```

### Configuration File

The `config.yaml` file contains all the parameters needed to run the trading system:

#### Market Settings
- `symbol`: Trading pair or stock symbol (e.g., "DOGE-USD.CC", "NVDA.US")
- `is_stock`: Set to true for stocks, false for crypto
- `crypto_pair_switch`: Enable crypto pair trading

#### Timeframe Settings
- `start_date`: Beginning of backtest period
- `end_date`: End of backtest period
- `timezone`: Timezone for data
- `interval1`: Daily interval (e.g., "1d")
- `interval2`: Intraday interval (e.g., "15m")
- `pivot_lookback`: Days to look back for pivot calculation

#### Grid Parameters
- `grid_quantized_level`: Grid spacing percentage
- `uniform_alloc_flag`: Uniform grid (1) or non-uniform (0)
- Support/Resistance levels for grid allocation
- Grid step percentages
- Rebalancing parameters

#### Strategy Parameters
- `interest_hk`: Interest flag for HighKick strategy
- `interest_ax`: Interest flag for AxeKick strategy
- `recycle_profit_ax_to_hk`: Recycle profits between strategies
- `maximize_base`/`maximize_quote`: Strategy optimization options

#### Capital Allocation
- `initial_money`: Total starting capital
- `initial_quote_hk_ratio`: Ratio of capital allocated to HighKick
- `initial_base_hk`: Initial base currency for HighKick

> **Important Note**: The `Initial_Base_AX` parameter is calculated dynamically in the script based on the price data, not directly from the config file. This ensures consistency with the original calculation method.

#### Trading Costs
- `slippage_buy`/`slippage_sell`: Trading slippage percentages
- `additional_trade_fee_margin`: Additional fees

#### API Credentials
- `eohd_api_key`: Your EOHD API key (required for data retrieval)

## How It Works

The system uses a config loader module that reads the YAML file and converts the settings into the format expected by the trading algorithms. This makes it easy to:

1. Run multiple strategy configurations without code changes
2. Save and load different parameter sets
3. Share configurations with others

## Output

The system generates:
- Trade history CSV file in the `results` directory
- Interactive charts showing:
  - Price action
  - Strategy performance
  - Support/Resistance levels
  - Rebalancing points

## Advanced Configuration

For advanced users, you can modify the `config_loader.py` file to add additional parameters or customize how they're processed.

## Note on API Key

You must provide a valid EOHD API key in the config file to download price data.