#!/usr/bin/env python3
"""
Bulk Import CSV Files to Portfolio Database
Automatically imports all CSV files from a directory into the portfolio database.
Asset names are derived from the first 11 characters of filenames.
"""

import sqlite3
import pandas as pd
import os
import glob
from datetime import datetime

# Configuration
DB_FILE = 'portfolio_data.db'
CSV_DIRECTORY = '/Users/samnazari/sandbox/atlas-cedar/Downloads/'  # Hardcoded directory

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
    print("✓ Multi-asset table created/verified")

def import_csv_file(csv_file):
    """
    Import a single CSV file into the database.
    
    Parameters:
    - csv_file: Path to CSV file
    
    Returns:
    - Tuple of (asset_name, records_imported, success)
    """
    try:
        # Extract asset name from filename (first 11 characters)
        filename = os.path.basename(csv_file)
        asset_name = filename[:11]
        
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Validate required columns
        if 'timestamp' not in df.columns:
            print(f"  ✗ Error: 'timestamp' column not found in {filename}")
            return (asset_name, 0, False)
        
        if 'portfolio_value' not in df.columns:
            print(f"  ✗ Error: 'portfolio_value' column not found in {filename}")
            return (asset_name, 0, False)
        
        if 'price' not in df.columns:
            print(f"  ✗ Error: 'price' column not found in {filename}")
            return (asset_name, 0, False)
        
        # Parse dates and values
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['portfolio_value'] = df['portfolio_value'].astype(float)
        df['price'] = df['price'].astype(float)
        
        # Prepare data for insertion
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Get count before insert
        cursor.execute("SELECT COUNT(*) FROM portfolio_data WHERE asset_name = ?", (asset_name,))
        before_count = cursor.fetchone()[0]
        
        # Insert data
        inserted = 0
        for _, row in df.iterrows():
            try:
                # Convert timestamp to string for SQLite
                timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                
                # current_value = portfolio_value (strategy)
                # initial_value = price (market price)
                cursor.execute('''
                    INSERT OR IGNORE INTO portfolio_data 
                    (asset_name, timestamp, current_value, initial_value, fee)
                    VALUES (?, ?, ?, ?, ?)
                ''', (asset_name, timestamp_str, row['portfolio_value'], row['price'], 0))
                if cursor.rowcount > 0:
                    inserted += 1
            except Exception as e:
                print(f"  ⚠ Error inserting row: {e}")
        
        conn.commit()
        
        # Get count after insert
        cursor.execute("SELECT COUNT(*) FROM portfolio_data WHERE asset_name = ?", (asset_name,))
        after_count = cursor.fetchone()[0]
        
        conn.close()
        
        return (asset_name, inserted, True)
        
    except FileNotFoundError:
        print(f"  ✗ Error: File '{csv_file}' not found")
        return (asset_name, 0, False)
    except Exception as e:
        print(f"  ✗ Error importing {filename}: {e}")
        return (asset_name, 0, False)

def bulk_import():
    """Import all CSV files from the configured directory."""
    print("\n" + "="*70)
    print("BULK CSV IMPORT")
    print("="*70)
    print(f"\nDirectory: {CSV_DIRECTORY}")
    
    # Check if directory exists
    if not os.path.exists(CSV_DIRECTORY):
        print(f"\n✗ Error: Directory '{CSV_DIRECTORY}' does not exist")
        return
    
    # Find all CSV files
    csv_pattern = os.path.join(CSV_DIRECTORY, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"\n✗ No CSV files found in {CSV_DIRECTORY}")
        return
    
    print(f"\nFound {len(csv_files)} CSV file(s)")
    print("-"*70)
    
    # Import each file
    results = []
    total_imported = 0
    
    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)
        print(f"\n▶ Processing: {filename}")
        
        asset_name, records_imported, success = import_csv_file(csv_file)
        results.append((filename, asset_name, records_imported, success))
        
        if success:
            print(f"  ✓ Imported {records_imported} records for asset '{asset_name}'")
            total_imported += records_imported
        else:
            print(f"  ✗ Failed to import {filename}")
    
    # Summary
    print("\n" + "="*70)
    print("IMPORT SUMMARY")
    print("="*70)
    
    successful = sum(1 for _, _, _, success in results if success)
    failed = len(results) - successful
    
    print(f"\nTotal files processed: {len(results)}")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"\nTotal records imported: {total_imported}")
    
    # Show asset breakdown
    print("\n" + "-"*70)
    print("Asset Breakdown:")
    print("-"*70)
    
    conn = sqlite3.connect(DB_FILE)
    query = """
        SELECT 
            asset_name,
            COUNT(*) as records,
            MIN(timestamp) as start_date,
            MAX(timestamp) as end_date,
            MIN(current_value) as min_value,
            MAX(current_value) as max_value,
            AVG(current_value) as avg_value
        FROM portfolio_data
        GROUP BY asset_name
        ORDER BY asset_name
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if not df.empty:
        for _, row in df.iterrows():
            print(f"\n{row['asset_name']}:")
            print(f"  Records: {row['records']}")
            print(f"  Date range: {row['start_date']} to {row['end_date']}")
            print(f"  Value range: ${row['min_value']:.2f} to ${row['max_value']:.2f}")
            print(f"  Average: ${row['avg_value']:.2f}")
    
    print("\n" + "="*70)
    print("✓ Bulk import completed!")
    print("="*70 + "\n")

def main():
    """Main entry point."""
    # Ensure multi-asset table exists
    create_multi_asset_table()
    
    # Run bulk import
    bulk_import()

if __name__ == "__main__":
    main()
