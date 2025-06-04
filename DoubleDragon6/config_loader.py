import os
import yaml

def load_config(config_file='config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_file (str): Path to the configuration file
        
    Returns:
        dict: Configuration parameters
    """
    normalized_config_file = os.path.normpath(config_file)

    if not os.path.exists(normalized_config_file):
        error_path_display = config_file if config_file == normalized_config_file else f"{config_file} (resolved to {normalized_config_file})"
        raise FileNotFoundError(f"Configuration file {error_path_display} not found.")
    
    with open(normalized_config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def get_trading_params(config):
    """
    Extract trading parameters from config in a format compatible with the trading script.
    
    Args:
        config (dict): Configuration dictionary loaded from YAML
        
    Returns:
        dict: Trading parameters ready to use in the script
    """
    params = {}
    
    # Market parameters
    params['symbol'] = config['market']['symbol']
    params['Stock_Flag'] = 1 if config['market']['is_stock'] else 0
    params['crypto_pair_switch'] = config['market']['crypto_pair_switch']
    params['results_dir'] = config['market'].get('results_dir', 'DoubleDragon6/results/')
    
    # Timeframe parameters
    params['start_date'] = config['timeframe']['start_date']
    params['end_date'] = config['timeframe']['end_date']
    params['timezone'] = config['timeframe']['timezone']
    params['interval1'] = config['timeframe']['interval1']
    params['interval2'] = config['timeframe']['interval2']
    params['pivot_lookback'] = config['timeframe']['pivot_lookback']
    
    # Grid parameters
    params['grid_quantized_level'] = config['grid']['grid_quantized_level']
    params['uniform_alloc_flag'] = config['grid']['uniform_alloc_flag']
    params['Support_BW'] = config['grid']['support_bandwidth']
    params['Spread_BW'] = config['grid']['spread_bandwidth']
    params['centralized_conservative_alloc'] = config['grid']['centralized_conservative_alloc']
    params['S_Level_HK'] = config['grid']['s_level_hk']
    params['R_Level_HK'] = config['grid']['r_level_hk']
    params['S_Level_AX'] = config['grid']['s_level_ax']
    params['R_Level_AX'] = config['grid']['r_level_ax']
    params['Step_Percent_hk'] = config['grid']['step_percent_hk']
    params['Step_Percent_ax'] = config['grid']['step_percent_ax']
    params['check_rebalance_offgrid'] = config['grid']['check_rebalance_offgrid']
    params['Reset_Grid_State'] = config['grid']['reset_grid_state']
    
    # Strategy parameters
    params['Interest_hk'] = config['strategy']['interest_hk']
    params['Interest_ax'] = config['strategy']['interest_ax']
    params['Recycle_Profit_AX_TO_HK'] = config['strategy']['recycle_profit_ax_to_hk']
    params['maximize_base'] = config['strategy']['maximize_base']
    params['maximize_quote'] = config['strategy']['maximize_quote']
    
    # Capital parameters
    params['Initial_Money'] = config['capital']['initial_money']
    params['Initial_Quote_HK'] = config['capital']['initial_money'] * config['capital']['initial_quote_hk_ratio']
    params['Initial_Base_HK'] = config['capital']['initial_base_hk']
    params['Initial_Quote_AX'] = 0
    # NOTE: Initial_Base_AX is pre-calculated here as a dollar amount,
    # but will be recalculated as a quantity using price data in the main script
    params['Initial_Base_AX'] = config['capital']['initial_money'] * (1 - config['capital']['initial_quote_hk_ratio'])
    
    # Cost parameters
    params['slippage_buy'] = config['costs']['slippage_buy']
    params['slippage_sell'] = config['costs']['slippage_sell']
    params['additional_trade_fee_margin'] = config['costs']['additional_trade_fee_margin']
    
    # API parameters
    params['eohd_api_key'] = config['api']['eohd_api_key']
    
    # Debug parameters
    params['trigger'] = config['debug']['trigger']
    
    return params

if __name__ == "__main__":
    # Test loading configuration
    config = load_config()
    params = get_trading_params(config)
    
    # Print loaded parameters
    for key, value in params.items():
        print(f"{key}: {value}") 