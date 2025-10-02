#!/bin/bash

# ═══════════════════════════════════════════════════════════════════════════
# 🚀 SPIKE DATA DOWNLOADER & DATABASE LOADER
# ═══════════════════════════════════════════════════════════════════════════

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

# Configuration - EDIT THESE VALUES
REMOTE_HOST="5.161.250.28"      # Your SSH server hostname/IP
REMOTE_USER="root"              # Your SSH username
REMOTE_PORT="22"                # SSH port (default is 22)
SSH_KEY="/Users/samnazari/sandbox/atlas-cedar/hetzner2.key"  # SSH private key
LOCAL_DOWNLOAD_DIR="/Users/samnazari/sandbox/atlas-cedar/Downloads/"  # Where to save files locally
DATABASE_PATH="/Users/samnazari/sandbox/atlas-cedar/portfolio_data.db"

# List of files to download
FILES_TO_DOWNLOAD=(
    "~/CoinBaseRun/BTC_Spike_Main/BTC_SPIKE_1_Output.csv"
    "~/CoinBaseRun/BTC_Spike_Main/BTC_SPIKE_2_Output.csv"
    "~/CoinBaseRun/ETH_Spike_Main/ETH_SPIKE_1_Output.csv"
    "~/CoinBaseRun/XRP_Spike_Main/XRP_SPIKE_1_Output.csv"
    "~/CoinBaseRun/SOL_Spike_Main/SOL_SPIKE_1_Output.csv"
)

# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

print_header() {
    echo -e "\n${CYAN}${BOLD}╔════════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${CYAN}${BOLD}║${RESET}  $1"
    echo -e "${CYAN}${BOLD}╚════════════════════════════════════════════════════════════════╝${RESET}\n"
}

print_step() {
    echo -e "${BLUE}▶${RESET} ${WHITE}$1${RESET}"
}

print_success() {
    echo -e "${GREEN}✓${RESET} $1"
}

print_error() {
    echo -e "${RED}✗${RESET} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${RESET} ${DIM}$1${RESET}"
}

print_separator() {
    echo -e "${DIM}────────────────────────────────────────────────────────────────${RESET}"
}

# ═══════════════════════════════════════════════════════════════════════════
# Main Script
# ═══════════════════════════════════════════════════════════════════════════

clear
echo -e "${MAGENTA}${BOLD}"
cat << "EOF"
   _____ ____ ___ __ __ ______   ____  ___  ______ ___ 
  / ___// __ \__ \/ //_// ____/  / __ \/   |/_  __//   |
  \__ \/ /_/ / / / ,<  / __/    / / / / /| | / /  / /| |
 ___/ / ____/ / / /| |/ /___   / /_/ / ___ |/ /  / ___ |
/____/_/   /_/_/_/ |_/_____/  /_____/_/  |_/_/  /_/  |_|
                                                         
EOF
echo -e "${RESET}"

print_header "📡 DOWNLOADING SPIKE DATA FILES"

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DOWNLOAD_DIR"
print_info "Download directory: $LOCAL_DOWNLOAD_DIR"
print_info "Remote server: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT}"
echo ""

# Check if sshpass is available for password auth
if command -v sshpass &> /dev/null; then
    print_step "Enter SSH password (or press Enter to use key only):"
    read -s SSH_PASSWORD
    echo ""
    if [ -n "$SSH_PASSWORD" ]; then
        export SSHPASS="$SSH_PASSWORD"
        USE_PASSWORD=true
        print_info "Using password authentication"
    else
        USE_PASSWORD=false
        print_info "Using SSH key authentication"
    fi
else
    USE_PASSWORD=false
    print_info "Using SSH key authentication (install sshpass for password option)"
fi
echo ""

# Download counter
total_files=${#FILES_TO_DOWNLOAD[@]}
current_file=0
success_count=0
failed_count=0

# Download each file
for remote_path in "${FILES_TO_DOWNLOAD[@]}"; do
    ((current_file++))
    base_name=$(basename "$remote_path")
    local_dest="$LOCAL_DOWNLOAD_DIR/$base_name"
    
    print_step "[$current_file/$total_files] Downloading: ${CYAN}$base_name${RESET}"
    
    if [ "$USE_PASSWORD" = true ]; then
        sshpass -e scp -o StrictHostKeyChecking=no -P "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST:$remote_path" "$local_dest" 2>&1
    else
        scp -o StrictHostKeyChecking=no -i "$SSH_KEY" -P "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST:$remote_path" "$local_dest" 2>&1
    fi
    
    if [ $? -eq 0 ]; then
        file_size=$(ls -lh "$local_dest" | awk '{print $5}')
        print_success "Downloaded ${BOLD}$base_name${RESET} (${file_size})"
        ((success_count++))
    else
        print_error "Failed to download $base_name"
        ((failed_count++))
    fi
    print_separator
done

echo ""
if [ $failed_count -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✅ All $total_files files downloaded successfully!${RESET}\n"
else
    echo -e "${YELLOW}${BOLD}⚠️  Downloaded: $success_count | Failed: $failed_count${RESET}\n"
fi

/opt/anaconda3/envs/gt/bin/python bulk_import_csv.py
/opt/anaconda3/envs/gt/bin/streamlit run streamlit_results_viewer.py
# ═══════════════════════════════════════════════════════════════════════════
# Database Loading
# ═══════════════════════════════════════════════════════════════════════════

# print_header "💾 LOADING DATA INTO DATABASE"

#print_step "Appending to database: ${CYAN}portfolio_data.db${RESET}"

# Get current record count before appending
#existing_records=$(sqlite3 "$DATABASE_PATH" "SELECT COUNT(*) FROM portfolio_data;" 2>/dev/null)
#if [ $? -eq 0 ]; then
#    print_info "Existing records in database: ${BOLD}$existing_records${RESET}"
#else
#    print_error "Failed to read database"
#fi
#print_separator

# Function to extract asset name from filename
#get_asset_name() {
#    local filename=$1
#    if [[ $filename == BTC_* ]]; then
#        echo "BTC"
#    elif [[ $filename == ETH_* ]]; then
#        echo "ETH"
#    elif [[ $filename == XRP_* ]]; then
#        echo "XRP"
#    elif [[ $filename == SOL_* ]]; then
#        echo "SOL"
#    else
#        echo "UNKNOWN"
#    fi
#}

# Load CSV files into database
#loaded_count=0
#new_records_added=0
#for csv_file in "$LOCAL_DOWNLOAD_DIR"/*.csv; do
#    if [ -f "$csv_file" ]; then
#        filename=$(basename "$csv_file")
#        asset_name=$(get_asset_name "$filename")
        
#        print_step "Loading: ${CYAN}$filename${RESET} → ${MAGENTA}$asset_name${RESET}"
        
        # Count lines (excluding header)
        #line_count=$(($(wc -l < "$csv_file") - 1))
        
        # Get count before insert
        #before_count=$(sqlite3 "$DATABASE_PATH" "SELECT COUNT(*) FROM portfolio_data;")
        
        # Import CSV into database
        # CSV columns: timestamp, price, portfolio_value
        # DB columns: asset_name, timestamp, current_value, initial_value, fee
        #sqlite3 "$DATABASE_PATH" <<EOF
#CREATE TEMP TABLE temp_import (
#    timestamp TEXT,
#    price REAL,
#    portfolio_value REAL
#);
#mode csv
#import --skip 1 "$csv_file" temp_import
#INSERT OR IGNORE INTO portfolio_data (asset_name, timestamp, current_value, initial_value, fee)
#SELECT '$asset_name', timestamp, price, portfolio_value, 0 FROM temp_import;
#DROP TABLE temp_import;
#EOF
        
        #if [ $? -eq 0 ]; then
            # Get count after insert
            #after_count=$(sqlite3 "$DATABASE_PATH" "SELECT COUNT(*) FROM portfolio_data;")
            #records_added=$((after_count - before_count))
            #new_records_added=$((new_records_added + records_added))
            
            #print_success "Processed ${BOLD}$line_count${RESET} records, added ${BOLD}$records_added${RESET} new (${BOLD}$((line_count - records_added))${RESET} duplicates skipped)"
            #((loaded_count++))
        #else
            #print_error "Failed to load $filename"
        #fi
        #print_separator
#    fi
#done

#echo ""
#print_header "📊 DATABASE SUMMARY"

# Get database statistics
#total_records=$(sqlite3 "$DATABASE_PATH" "SELECT COUNT(*) FROM portfolio_data;")
#btc_records=$(sqlite3 "$DATABASE_PATH" "SELECT COUNT(*) FROM portfolio_data WHERE asset_name='BTC';")
#eth_records=$(sqlite3 "$DATABASE_PATH" "SELECT COUNT(*) FROM portfolio_data WHERE asset_name='ETH';")
#xrp_records=$(sqlite3 "$DATABASE_PATH" "SELECT COUNT(*) FROM portfolio_data WHERE asset_name='XRP';")
#sol_records=$(sqlite3 "$DATABASE_PATH" "SELECT COUNT(*) FROM portfolio_data WHERE asset_name='SOL';")

#echo -e "${BOLD}Total Records:${RESET} ${GREEN}$total_records${RESET} ${DIM}(+$new_records_added new)${RESET}"
#echo -e "  ${YELLOW}●${RESET} BTC: $btc_records records"
#echo -e "  ${BLUE}●${RESET} ETH: $eth_records records"
#echo -e "  ${CYAN}●${RESET} XRP: $xrp_records records"
#echo -e "  ${MAGENTA}●${RESET} SOL: $sol_records records"

#echo ""
#echo -e "${GREEN}${BOLD}🎉 ALL OPERATIONS COMPLETED SUCCESSFULLY!${RESET}\n"
