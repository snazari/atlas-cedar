#!/usr/bin/env python3
"""
Import Asset Data from CSV to Portfolio Database
This script allows importing additional asset data from CSV files.
"""

import sqlite3
import pandas as pd
from datetime import datetime
import sys

DB_FILE = 'portfolio_data.db'

def create_multi_asset_table():
    """Create a new table that supports multiple assets."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create new table for multiple assets if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_name TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            current_value REAL NOT NULL,
            initial_value REAL,
            fee REAL,
            UNIQUE(asset_name, timestamp)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Multi-asset table created/verified")

def migrate_btc_data():
    """Migrate existing BTC data to the new multi-asset table."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Check if migration is needed
    cursor.execute("SELECT COUNT(*) FROM portfolio_data WHERE asset_name = 'BTC-USD'")
    if cursor.fetchone()[0] > 0:
        print("BTC-USD data already migrated")
        conn.close()
        return
    
    # Migrate data from btc_balance to portfolio_data
    cursor.execute('''
        INSERT OR IGNORE INTO portfolio_data (asset_name, timestamp, current_value, initial_value, fee)
        SELECT 'BTC-USD', timestamp, current_value, initial_value, fee
        FROM btc_balance
    ''')
    
    migrated = cursor.rowcount
    conn.commit()
    conn.close()
    print(f"Migrated {migrated} BTC-USD records to multi-asset table")

def import_csv_data(csv_file, asset_name, date_column='Date', value_column='Value'):
    """
    Import asset data from CSV file.
    
    Parameters:
    - csv_file: Path to CSV file
    - asset_name: Name of the asset (e.g., 'SPY', 'ETH-USD')
    - date_column: Name of the date column in CSV
    - value_column: Name of the value column in CSV
    """
    try:
        # Read CSV
        df = pd.read_csv(csv_file)
        print(f"Read {len(df)} rows from {csv_file}")
        
        # Show first few rows and columns to help user
        print("\nFirst 3 rows of your CSV:")
        print(df.head(3))
        print(f"\nColumns found: {df.columns.tolist()}")
        
        # Check if specified columns exist
        if date_column not in df.columns:
            print(f"\nError: Column '{date_column}' not found in CSV")
            print(f"Available columns: {df.columns.tolist()}")
            return
        
        if value_column not in df.columns:
            print(f"\nError: Column '{value_column}' not found in CSV")
            print(f"Available columns: {df.columns.tolist()}")
            return
        
        # Parse dates
        df['timestamp'] = pd.to_datetime(df[date_column])
        df['current_value'] = df[value_column].astype(float)
        
        # Get initial value (first value in the series)
        initial_value = df['current_value'].iloc[0]
        
        # Prepare data for insertion
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Insert data
        inserted = 0
        for _, row in df.iterrows():
            try:
                # Convert timestamp to string for SQLite
                timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute('''
                    INSERT OR IGNORE INTO portfolio_data 
                    (asset_name, timestamp, current_value, initial_value, fee)
                    VALUES (?, ?, ?, ?, ?)
                ''', (asset_name, timestamp_str, row['current_value'], initial_value, 0))
                if cursor.rowcount > 0:
                    inserted += 1
            except Exception as e:
                print(f"Error inserting row: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"\nSuccessfully imported {inserted} records for {asset_name}")
        
        # Show summary
        conn = sqlite3.connect(DB_FILE)
        summary = pd.read_sql_query(f"""
            SELECT 
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date,
                COUNT(*) as record_count,
                MIN(current_value) as min_value,
                MAX(current_value) as max_value,
                AVG(current_value) as avg_value
            FROM portfolio_data
            WHERE asset_name = '{asset_name}'
        """, conn)
        conn.close()
        
        print(f"\nSummary for {asset_name}:")
        print(f"  Date range: {summary['start_date'].iloc[0]} to {summary['end_date'].iloc[0]}")
        print(f"  Records: {summary['record_count'].iloc[0]}")
        print(f"  Value range: ${summary['min_value'].iloc[0]:.2f} to ${summary['max_value'].iloc[0]:.2f}")
        print(f"  Average: ${summary['avg_value'].iloc[0]:.2f}")
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
    except Exception as e:
        print(f"Error importing CSV: {e}")

def list_assets():
    """List all assets in the database."""
    conn = sqlite3.connect(DB_FILE)
    query = """
        SELECT 
            asset_name,
            COUNT(*) as records,
            MIN(timestamp) as start_date,
            MAX(timestamp) as end_date,
            MIN(current_value) as min_value,
            MAX(current_value) as max_value
        FROM portfolio_data
        GROUP BY asset_name
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("No assets found in database")
    else:
        print("\nAssets in database:")
        print(df.to_string(index=False))

def interactive_import():
    """Interactive CSV import wizard."""
    print("\n" + "="*60)
    print("CSV Import Wizard")
    print("="*60)
    
    # Get CSV file
    csv_file = input("\nEnter path to CSV file: ").strip()
    if not csv_file:
        print("No file specified")
        return
    
    # Try to read and show the CSV structure
    try:
        df = pd.read_csv(csv_file)
        print(f"\nFound {len(df)} rows and {len(df.columns)} columns")
        print("\nFirst 5 rows:")
        print(df.head())
        print(f"\nColumns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Get asset name
    asset_name = input("\nEnter asset name (e.g., SPY, ETH-USD, GOLD): ").strip()
    if not asset_name:
        print("No asset name specified")
        return
    
    # Get column names
    print("\nWhich column contains the dates?")
    date_column = input(f"Date column name (default: '{df.columns[0]}'): ").strip() or df.columns[0]
    
    print("\nWhich column contains the portfolio values?")
    # Try to guess the value column
    value_cols = [col for col in df.columns if 'value' in col.lower() or 'price' in col.lower() or 'close' in col.lower()]
    default_val_col = value_cols[0] if value_cols else df.columns[1]
    value_column = input(f"Value column name (default: '{default_val_col}'): ").strip() or default_val_col
    
    # Confirm
    print(f"\nReady to import:")
    print(f"  File: {csv_file}")
    print(f"  Asset: {asset_name}")
    print(f"  Date column: {date_column}")
    print(f"  Value column: {value_column}")
    
    confirm = input("\nProceed? (y/n): ")
    if confirm.lower() == 'y':
        import_csv_data(csv_file, asset_name, date_column, value_column)

def main():
    """Main menu."""
    # Ensure multi-asset table exists
    create_multi_asset_table()
    
    # Check if we need to migrate BTC data
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='btc_balance'")
    if cursor.fetchone():
        migrate_btc_data()
    conn.close()
    
    if len(sys.argv) > 1:
        # Command line mode
        if len(sys.argv) < 3:
            print("Usage: python import_asset_data.py <csv_file> <asset_name> [date_column] [value_column]")
            print("Example: python import_asset_data.py spy_data.csv SPY Date Close")
            sys.exit(1)
        
        csv_file = sys.argv[1]
        asset_name = sys.argv[2]
        date_column = sys.argv[3] if len(sys.argv) > 3 else 'Date'
        value_column = sys.argv[4] if len(sys.argv) > 4 else 'Value'
        
        import_csv_data(csv_file, asset_name, date_column, value_column)
    else:
        # Interactive mode
        while True:
            print("\n" + "="*60)
            print("Multi-Asset Portfolio Database Manager")
            print("="*60)
            print("1. Import CSV data (interactive)")
            print("2. List all assets")
            print("3. Exit")
            
            choice = input("\nEnter choice (1-3): ")
            if choice == '1':
                interactive_import()
            elif choice == '2':
                list_assets()
            elif choice == '3':
                print("Goodbye!")
                break
            else:
                print("Invalid choice")

if __name__ == "__main__":
    main()
