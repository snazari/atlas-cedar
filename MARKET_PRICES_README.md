# Market Price Import System

## 📊 Overview

The system automatically imports market prices directly from CSV files along with portfolio data to enable proper beta analysis and performance comparisons.

## 🔄 How It Works

### 1. **Automated Workflow** (`get_latest_spike.sh`)
```bash
1. Download CSV files from remote server
2. Import portfolio data AND market prices to database (bulk_import_csv.py)
3. Launch Streamlit dashboard
```

### 2. **CSV File Structure**

Each CSV file must contain these columns:
- **`timestamp`** - Date/time of the record
- **`portfolio_value`** - Strategy/portfolio value (stored as `current_value`)
- **`price`** - Market price of the asset (stored as `initial_value`)

#### Example CSV:
```csv
timestamp,portfolio_value,price
2024-01-01 12:00:00,5000.00,42500.50
2024-01-01 12:05:00,5010.25,42510.75
...
```

### 3. **Import Process** (`bulk_import_csv.py`)

1. **Reads CSV files** from Downloads directory
2. **Validates columns** (timestamp, portfolio_value, price)
3. **Imports to database**:
   - `current_value` = `portfolio_value` (strategy)
   - `initial_value` = `price` (market price)
4. **Reports statistics** for verification

- ✅ Automatic column validation
- ✅ Duplicate prevention (INSERT OR IGNORE)
- ✅ Comprehensive error handling
- ✅ Verification report after completion

## 📊 Database Schema

### After CSV Import
```sql
portfolio_data:
  - asset_name: "BTC_SPIKE_1"
  - timestamp: "2024-01-01 12:00:00"
  - current_value: 5000.00   (portfolio_value from CSV)
  - initial_value: 42500.50  (price from CSV)
  - fee: 0.00
```

### Column Mapping
- **`current_value`** = Strategy/Portfolio value (`portfolio_value` from CSV)
- **`initial_value`** = Market price (`price` from CSV)
- **`timestamp`** = Date/time of record
- **`asset_name`** = First 11 characters of filename
- **`fee`** = Transaction fee (default: 0)

## 🎯 Beta Analysis Benefits

With market prices from CSV:

1. **Accurate Beta Calculation**
   - Strategy returns vs Market returns
   - Proper CAPM analysis

2. **Alpha Metrics**
   - Regular Alpha
   - Jensen's Alpha
   - Regression Alpha

3. **Performance Ratios**
   - Sharpe Ratio
   - Sortino Ratio
   - Treynor Ratio
   - Information Ratio

4. **Visualizations**
   - Scatter plots (strategy vs market)
   - Cumulative return comparisons
   - Risk-return profiles
   - Drawdown analysis

## 🚀 Usage

### Automatic (Recommended)
```bash
./get_latest_spike.sh
```
This runs the complete workflow:
1. Downloads CSV files from server
2. Imports portfolio data AND market prices
3. Launches dashboard

### Manual Import
```bash
# Just import CSV files to database
python bulk_import_csv.py
```

## 📊 Verification

The import script provides a verification report:

```
Asset Breakdown:
----------------------------------------------------------------------

BTC_SPIKE_1:
  Records: 1000
  Date range: 2024-01-01 00:00:00 to 2024-12-31 23:55:00
  Value range: $4,500.00 to $5,500.00
  Average: $5,000.00
```

## 🐛 Troubleshooting

### Issue: Missing 'price' column
**Solution**: Ensure CSV files have `timestamp`, `portfolio_value`, and `price` columns

### Issue: No data imported
**Solution**: Check CSV file format and column names (case-sensitive)

### Issue: Duplicate data
**Solution**: Script uses INSERT OR IGNORE - duplicates are automatically skipped

## 📝 Notes

- Market prices are read directly from CSV `price` column
- Portfolio values are read from CSV `portfolio_value` column
- Market prices stored in database `initial_value` column
- Portfolio values stored in database `current_value` column
- Asset names derived from first 11 characters of filename
- All prices assumed to be in USD

## 🔗 Related Files

- `get_latest_spike.sh` - Main automation script
- `bulk_import_csv.py` - CSV import to database (includes market prices)
- `streamlit_results_viewer.py` - Dashboard (Tab 5: Beta Analysis)
- `beta_analyzer_streamlit.py` - Beta calculation engine

## ✅ Summary

The system now correctly uses:
- **CSV `price` column** → Market price (for beta calculation)
- **CSV `portfolio_value` column** → Strategy value (for performance tracking)

No external API calls needed - all data comes from your CSV files!
