#!/usr/bin/env python3
"""
Database Manager for Portfolio Data
This script provides utilities to view, edit, and manage the portfolio database.
"""

import sqlite3
import pandas as pd
from datetime import datetime
import sys

DB_FILE = 'portfolio_data.db'

def view_data(limit=10):
    """View the most recent records in the database."""
    conn = sqlite3.connect(DB_FILE)
    query = f"SELECT * FROM btc_balance ORDER BY timestamp DESC LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def remove_duplicates():
    """Remove duplicate entries based on timestamp, keeping the first occurrence."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # First, let's see how many duplicates we have
    cursor.execute("""
        SELECT timestamp, COUNT(*) as count 
        FROM btc_balance 
        GROUP BY timestamp 
        HAVING COUNT(*) > 1
    """)
    duplicates = cursor.fetchall()
    
    if duplicates:
        print(f"Found {len(duplicates)} timestamps with duplicates")
        
        # Delete duplicates, keeping the one with the lowest ID for each timestamp
        cursor.execute("""
            DELETE FROM btc_balance 
            WHERE id NOT IN (
                SELECT MIN(id) 
                FROM btc_balance 
                GROUP BY timestamp
            )
        """)
        
        deleted = cursor.rowcount
        conn.commit()
        print(f"Removed {deleted} duplicate entries")
    else:
        print("No duplicates found")
    
    conn.close()

def update_value(timestamp_str, field, new_value):
    """Update a specific field for a specific timestamp."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Validate field name
    valid_fields = ['current_value', 'initial_value', 'fee']
    if field not in valid_fields:
        print(f"Error: field must be one of {valid_fields}")
        return
    
    cursor.execute(f"""
        UPDATE btc_balance 
        SET {field} = ? 
        WHERE timestamp = ?
    """, (new_value, timestamp_str))
    
    if cursor.rowcount > 0:
        conn.commit()
        print(f"Updated {cursor.rowcount} record(s)")
    else:
        print("No records found with that timestamp")
    
    conn.close()

def delete_by_date_range(start_date, end_date):
    """Delete records within a date range."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        DELETE FROM btc_balance 
        WHERE timestamp BETWEEN ? AND ?
    """, (start_date, end_date))
    
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    
    print(f"Deleted {deleted} records between {start_date} and {end_date}")

def get_stats():
    """Get statistics about the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM btc_balance")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM btc_balance")
    min_date, max_date = cursor.fetchone()
    
    cursor.execute("SELECT AVG(current_value), MIN(current_value), MAX(current_value) FROM btc_balance")
    avg_val, min_val, max_val = cursor.fetchone()
    
    conn.close()
    
    print(f"Database Statistics:")
    print(f"  Total records: {total}")
    print(f"  Date range: {min_date} to {max_date}")
    print(f"  Portfolio value range: ${min_val:.2f} to ${max_val:.2f}")
    print(f"  Average portfolio value: ${avg_val:.2f}")

def export_to_csv(filename="portfolio_export.csv"):
    """Export all data to CSV."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM btc_balance ORDER BY timestamp", conn)
    conn.close()
    
    df.to_csv(filename, index=False)
    print(f"Exported {len(df)} records to {filename}")

def import_from_csv(filename):
    """Import data from CSV (be careful - this adds to existing data)."""
    df = pd.read_csv(filename)
    
    conn = sqlite3.connect(DB_FILE)
    df.to_sql('btc_balance', conn, if_exists='append', index=False)
    conn.close()
    
    print(f"Imported {len(df)} records from {filename}")

def main():
    """Main menu for database management."""
    while True:
        print("\n" + "="*50)
        print("Portfolio Database Manager")
        print("="*50)
        print("1. View recent data")
        print("2. Remove duplicates")
        print("3. Get database statistics")
        print("4. Export to CSV")
        print("5. Update a specific value")
        print("6. Delete records by date range")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ")
        
        if choice == '1':
            limit = input("How many records to view? (default 10): ") or "10"
            df = view_data(int(limit))
            print("\n", df.to_string())
            
        elif choice == '2':
            confirm = input("Remove all duplicate entries? (y/n): ")
            if confirm.lower() == 'y':
                remove_duplicates()
                
        elif choice == '3':
            get_stats()
            
        elif choice == '4':
            filename = input("Enter filename (default: portfolio_export.csv): ") or "portfolio_export.csv"
            export_to_csv(filename)
            
        elif choice == '5':
            timestamp = input("Enter timestamp (YYYY-MM-DD HH:MM:SS): ")
            field = input("Enter field to update (current_value/initial_value/fee): ")
            value = float(input("Enter new value: "))
            update_value(timestamp, field, value)
            
        elif choice == '6':
            start = input("Enter start date (YYYY-MM-DD HH:MM:SS): ")
            end = input("Enter end date (YYYY-MM-DD HH:MM:SS): ")
            confirm = input(f"Delete all records between {start} and {end}? (y/n): ")
            if confirm.lower() == 'y':
                delete_by_date_range(start, end)
                
        elif choice == '7':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
