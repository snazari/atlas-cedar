# Atlas Cedar Gmail Integration

## 1. Overview

This document provides instructions for setting up and using the new Gmail integration feature for the Atlas Cedar Backtest Results Viewer. This feature adds a real-time monitoring dashboard that tracks a live BTC-USD portfolio by fetching and parsing data from Gmail emails.

## 2. Prerequisites

Before you begin, ensure you have the following:
- Python 3.8+ installed.
- Access to a Google Account.

## 3. Setup Instructions

### 3.1. Install Dependencies

Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

### 3.2. Enable Gmail API and Get Credentials

1.  **Go to the Google Cloud Console**: [https://console.cloud.google.com/](https://console.cloud.google.com/)
2.  **Create a new project** or select an existing one.
3.  **Enable the Gmail API**:
    -   In the navigation menu, go to "APIs & Services" > "Library".
    -   Search for "Gmail API" and enable it.
4.  **Create credentials**:
    -   Go to "APIs & Services" > "Credentials".
    -   Click "Create Credentials" > "OAuth client ID".
    -   If prompted, configure the "OAuth consent screen". Choose "External" and provide a name for the application.
    -   For the "Application type", select "Desktop app".
    -   Click "Create". A dialog will appear with your client ID and client secret.
    -   **Download the JSON file** and save it as `credentials.json` in the root directory of this project.

## 4. Running the Application

The system consists of two main components: the background email fetching service and the Streamlit dashboard.

### 4.1. Run the Email Fetching Service

This service runs continuously to check for new emails and update the local database.

1.  **Open a terminal** and navigate to the project directory.
2.  **Run the script**:
    ```bash
    python gmail_integration.py
    ```
3.  **First-time Authentication**:
    -   On the first run, a browser window will open, prompting you to log in to your Google Account and grant permission for the application to read your emails.
    -   After you grant permission, a `token.json` file will be created in the project directory. This file will be used for authentication in subsequent runs.
4.  The script will then start the scheduler and check for emails every hour. You can leave this script running in the background.

### 4.2. View the Dashboard

1.  **Open another terminal** in the project directory.
2.  **Run the Streamlit app**:
    ```bash
    streamlit run streamlit_results_viewer.py
    ```
3.  A browser window will open with the application.
4.  Navigate to the **"Live BTC-USD Portfolio"** tab to see your dashboard.
5.  Click the **"Refresh Data"** button to load the latest data from the database.
