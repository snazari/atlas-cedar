#!/usr/bin/env python3
"""
Fetch Market Prices for Portfolio Assets
Downloads historical price data from CoinGecko API and updates the database
"""

import sqlite3
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import numpy as np

# Configuration
DB_FILE = 'portfolio_data.db'

# Asset mapping: portfolio asset name -> CoinGecko ID
ASSET_MAPPING = {
    'BTC_SPIKE_1': 'bitcoin',
    'BTC_SPIKE_2': 'bitcoin',
    'ETH_SPIKE_1': 'ethereum',
    'XRP_SPIKE_1': 'ripple',
    'SOL_SPIKE_1': 'solana'
}

def get_coingecko_prices(coin_id, start_timestamp, end_timestamp):
    """
    Fetch historical prices from CoinGecko API.
    
    Parameters:
    - coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
    - start_timestamp: Unix timestamp for start date
    - end_timestamp: Unix timestamp for end date
    
    Returns:
    - DataFrame with timestamp and price columns
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    
    params = {
        'vs_currency': 'usd',
        'from': start_timestamp,
        'to': end_timestamp
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract prices
        prices = data.get('prices', [])
        
        if not prices:
            print(f"  ⚠️  No price data returned for {coin_id}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(prices, columns=['timestamp_ms', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
        df = df[['timestamp', 'price']]
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error fetching data for {coin_id}: {e}")
        return pd.DataFrame()

def interpolate_prices(price_df, portfolio_df):
    """
    Interpolate market prices to match portfolio timestamps.
    
    Parameters:
    - price_df: DataFrame with market prices
    - portfolio_df: DataFrame with portfolio timestamps
    
    Returns:
    - Series with interpolated prices matching portfolio timestamps
    """
    # Merge on timestamp using nearest match
    portfolio_df = portfolio_df.sort_values('timestamp')
    price_df = price_df.sort_values('timestamp')
    
    # Use merge_asof for nearest timestamp matching
    merged = pd.merge_asof(
        portfolio_df[['timestamp']],
        price_df,
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('1 hour')  # Allow 1 hour tolerance
    )
    
    return merged['price']

def update_market_prices_for_asset(asset_name):
    """
    Fetch and update market prices for a specific asset.
    
    Parameters:
    - asset_name: Name of the asset in the database
    
    Returns:
    - Number of records updated
    """
    # Get CoinGecko ID
    coin_id = ASSET_MAPPING.get(asset_name)
    if not coin_id:
        print(f"  ⚠️  No CoinGecko mapping for {asset_name}")
        return 0
    
    print(f"\n▶ Processing {asset_name} ({coin_id})...")
    
    # Connect to database
    conn = sqlite3.connect(DB_FILE)
    
    # Get portfolio data for this asset
    query = """
        SELECT timestamp, current_value 
        FROM portfolio_data 
        WHERE asset_name = ?
        ORDER BY timestamp
    """
    portfolio_df = pd.read_sql_query(query, conn, params=(asset_name,))
    
    if portfolio_df.empty:
        print(f"  ⚠️  No data found for {asset_name}")
        conn.close()
        return 0
    
    # Parse timestamps
    portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
    
    # Get date range
    start_date = portfolio_df['timestamp'].min()
    end_date = portfolio_df['timestamp'].max()
    
    print(f"  📅 Date range: {start_date} to {end_date}")
    print(f"  📊 Portfolio records: {len(portfolio_df)}")
    
    # Convert to Unix timestamps
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())
    
    # Fetch market prices
    print(f"  🌐 Fetching market prices from CoinGecko...")
    price_df = get_coingecko_prices(coin_id, start_timestamp, end_timestamp)
    
    if price_df.empty:
        print(f"  ✗ Failed to fetch prices for {asset_name}")
        conn.close()
        return 0
    
    print(f"  ✓ Fetched {len(price_df)} price points")
    
    # Interpolate prices to match portfolio timestamps
    print(f"  🔄 Interpolating prices to match portfolio timestamps...")
    interpolated_prices = interpolate_prices(price_df, portfolio_df)
    
    # Update database
    print(f"  💾 Updating database...")
    cursor = conn.cursor()
    updated = 0
    
    for idx, row in portfolio_df.iterrows():
        price = interpolated_prices.iloc[idx]
        
        if pd.notna(price):
            timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''
                UPDATE portfolio_data 
                SET initial_value = ?
                WHERE asset_name = ? AND timestamp = ?
            ''', (price, asset_name, timestamp_str))
            
            if cursor.rowcount > 0:
                updated += 1
    
    conn.commit()
    conn.close()
    
    print(f"  ✓ Updated {updated} records with market prices")
    
    # Rate limiting for API
    time.sleep(1.5)  # CoinGecko free tier: ~50 calls/minute
    
    return updated

def verify_market_prices():
    """Verify that market prices have been added correctly."""
    conn = sqlite3.connect(DB_FILE)
    
    query = """
        SELECT 
            asset_name,
            COUNT(*) as total_records,
            SUM(CASE WHEN initial_value IS NOT NULL THEN 1 ELSE 0 END) as with_prices,
            MIN(initial_value) as min_price,
            MAX(initial_value) as max_price,
            AVG(initial_value) as avg_price
        FROM portfolio_data
        GROUP BY asset_name
        ORDER BY asset_name
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print("\n" + "="*70)
    print("MARKET PRICE VERIFICATION")
    print("="*70)
    
    for _, row in df.iterrows():
        print(f"\n{row['asset_name']}:")
        print(f"  Total records: {row['total_records']}")
        print(f"  Records with prices: {row['with_prices']}")
        
        if row['with_prices'] > 0:
            coverage = (row['with_prices'] / row['total_records']) * 100
            print(f"  Coverage: {coverage:.1f}%")
            print(f"  Price range: ${row['min_price']:.2f} to ${row['max_price']:.2f}")
            print(f"  Average price: ${row['avg_price']:.2f}")
        else:
            print(f"  ⚠️  No market prices found!")

def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("MARKET PRICE FETCHER")
    print("="*70)
    print("\nFetching historical market prices from CoinGecko API...")
    print("This may take a few minutes due to API rate limiting.\n")
    
    # Get all unique assets from database
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT asset_name FROM portfolio_data ORDER BY asset_name")
    assets = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    if not assets:
        print("✗ No assets found in database")
        return
    
    print(f"Found {len(assets)} asset(s): {', '.join(assets)}\n")
    
    # Update prices for each asset
    total_updated = 0
    for asset in assets:
        updated = update_market_prices_for_asset(asset)
        total_updated += updated
    
    # Verify results
    verify_market_prices()
    
    print("\n" + "="*70)
    print(f"✓ Market price fetch completed! Total records updated: {total_updated}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
