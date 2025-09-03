import os
import re
import sqlite3
import schedule
import time
import base64
from datetime import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# Constants
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
TOKEN_FILE = 'token.json'
CREDENTIALS_FILE = 'credentials.json'
DB_FILE = 'portfolio_data.db'
SENDER_EMAIL = 'mfarsh@gmail.com'
EMAIL_SUBJECT = 'dataframe email'

def init_db():
    """
    Initializes the SQLite database and creates the btc_balance table if it doesn't exist.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS btc_balance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            current_value REAL NOT NULL,
            initial_value REAL NOT NULL,
            fee REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_balance_data(timestamp, current_value, initial_value, fee):
    """
    Inserts a new record into the btc_balance table.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO btc_balance (timestamp, current_value, initial_value, fee)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, current_value, initial_value, fee))
    conn.commit()
    conn.close()

def get_email_body(payload):
    """
    Extracts the text/plain body from an email payload, handling multipart and single-part messages.
    """
    body = ""
    if 'parts' in payload:
        # It's a multipart message, find the text/plain part.
        part = next((p for p in payload['parts'] if p['mimeType'] == 'text/plain'), None)
        if part:
            data = part['body'].get('data')
            if data:
                body = base64.urlsafe_b64decode(data).decode('utf-8')
    elif 'body' in payload:
        # It's a single-part message.
        data = payload['body'].get('data')
        if data:
            body = base64.urlsafe_b64decode(data).decode('utf-8')
    return body

def fetch_and_process_emails(service):
    """
    Fetches unread emails from the specified sender and subject,
    parses them, and stores the data in the database.
    """
    # Search for emails containing 'dataframe' and 'email' in subject (case-insensitive)
    query = f"from:{SENDER_EMAIL} subject:dataframe subject:email is:unread"
    
    # Fetch all pages of results
    messages = []
    page_token = None
    
    while True:
        if page_token:
            results = service.users().messages().list(
                userId='me', q=query, pageToken=page_token, maxResults=500
            ).execute()
        else:
            results = service.users().messages().list(
                userId='me', q=query, maxResults=500
            ).execute()
        
        messages.extend(results.get('messages', []))
        
        page_token = results.get('nextPageToken')
        if not page_token:
            break
    
    if not messages:
        print("No new emails found.")
    else:
        print(f"Found {len(messages)} emails to process...")
        for i, message in enumerate(messages, 1):
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            
            # Get email timestamp
            internal_date = int(msg['internalDate']) / 1000
            timestamp = datetime.fromtimestamp(internal_date)

            payload = msg['payload']
            body = get_email_body(payload)

            if body:
                parsed_data = parse_email_body(body)
                if parsed_data:
                    insert_balance_data(
                        timestamp,
                        parsed_data['current_value'],
                        parsed_data['initial_value'],
                        parsed_data['fee']
                    )
                    print(f"[{i}/{len(messages)}] Processed email from {timestamp}")
                    # Mark email as read
                    service.users().messages().modify(
                        userId='me', id=message['id'], body={'removeLabelIds': ['UNREAD']}
                    ).execute()
                else:
                    print(f"[{i}/{len(messages)}] Failed to parse email from {timestamp}")
                    print("--- Email Body (first 500 chars) ---")
                    print(body[:500])
                    print("------------------------------------")
            else:
                print(f"[{i}/{len(messages)}] Could not find text/plain body for email from {timestamp}")

def parse_email_body(body):
    """
    Parses the email body to extract portfolio metrics using regular expressions.
    """
    pattern = re.compile(
        r"current portfolio.*?value is:\s*([\d\.]+)\s*"
        r"initial value is:\s*([\d\.]+)\s*"
        r"fee is:?\s*([\d\.]+)",
        re.IGNORECASE | re.DOTALL
    )
    
    match = pattern.search(body)
    
    if match:
        try:
            current_value = float(match.group(1))
            initial_value = float(match.group(2))
            fee = float(match.group(3))
            return {
                'current_value': current_value,
                'initial_value': initial_value,
                'fee': fee
            }
        except (ValueError, IndexError):
            return None
    return None

def get_gmail_service():
    """
    Authenticates with the Gmail API and returns a service object.
    Handles the OAuth 2.0 flow.
    """
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

def main():
    """
    Main function to run the Gmail integration service.
    """
    init_db()
    service = get_gmail_service()
    print("Successfully connected to Gmail API.")
    
    # Schedule the job
    schedule.every().hour.do(fetch_and_process_emails, service=service)
    
    # Run the job immediately on startup
    fetch_and_process_emails(service)
    
    print("Scheduler started. Checking for emails every hour.")
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    main()
