import os
import psycopg2
from datetime import date, timedelta
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
import json
import time
import argparse
from typing import Optional, Dict, List
import socket
import urllib.parse
from psycopg2.extras import RealDictCursor
import requests

load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")
SITE_URL = os.getenv("SITE_URL")
DATABASE_URL = os.getenv("DATABASE_URL")

# Supabase REST API settings
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
USE_REST_API = os.getenv("USE_REST_API", "false").lower() == "true"

class GSCDataFetcher:
    def __init__(self):
        self.service = None
        self.use_rest_api = USE_REST_API
        self.connect_to_gsc()
    
    def connect_to_gsc(self):
        """Initialize GSC connection"""
        try:
            creds = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=SCOPES
            )
            self.service = build('searchconsole', 'v1', credentials=creds, cache_discovery=False)
            print("✓ Connected to Google Search Console")
        except Exception as e:
            print(f"✗ Error connecting to GSC: {e}")
            raise
    
    def test_gsc_connection(self):
        """Test if we can connect to GSC and list sites"""
        try:
            sites = self.service.sites().list().execute()
            print("Sites accessible by service account:")
            for site in sites.get('siteEntry', []):
                print(f"  - {site['siteUrl']} (Permission: {site['permissionLevel']})")
            return True
        except Exception as e:
            print(f"Error testing GSC connection: {e}")
            return False
    
    def fetch_gsc_data(self, start_date: str, end_date: str) -> Optional[Dict]:
        """Fetch GSC data for a specific date range"""
        try:
            request = {
                "startDate": start_date,
                "endDate": end_date,
                "dimensions": ["date"],
                "rowLimit": 25000  # Max limit for GSC API
            }
            
            print(f"Fetching data for {start_date} to {end_date}")
            response = self.service.searchanalytics().query(siteUrl=SITE_URL, body=request).execute()
            
            if "rows" in response and len(response["rows"]) > 0:
                return response["rows"]
            else:
                print(f"No data found for {start_date} to {end_date}")
                return None
                
        except HttpError as e:
            print(f"HTTP Error: {e}")
            if e.resp.status == 403:
                print("Permission denied. Make sure the service account is added to GSC with proper permissions.")
            elif e.resp.status == 400:
                print("Bad request. Check site URL format and date range.")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    
    def get_db_connection(self, retries=3, timeout=30):
        """Get database connection with retry logic and IPv4 preference"""
        if self.use_rest_api:
            return None  # Skip PostgreSQL connection
            
        database_url = DATABASE_URL
        
        # Parse the URL to modify connection parameters
        parsed = urllib.parse.urlparse(database_url)
        
        # Build connection parameters
        conn_params = {
            'host': parsed.hostname,
            'port': parsed.port or 5432,
            'database': parsed.path.lstrip('/'),
            'user': parsed.username,
            'password': parsed.password,
            'connect_timeout': timeout,
            'application_name': 'gsc_fetcher_github_actions'
        }
        
        # Add SSL requirement for Supabase
        if 'supabase.co' in parsed.hostname:
            conn_params['sslmode'] = 'require'
        
        for attempt in range(retries):
            try:
                print(f"Connection attempt {attempt + 1}/{retries} to {conn_params['host']}:{conn_params['port']}")
                
                # Try to resolve hostname to IPv4 first
                try:
                    # Force IPv4 resolution
                    ip_address = socket.getaddrinfo(
                        conn_params['host'], 
                        conn_params['port'], 
                        socket.AF_INET,  # Force IPv4
                        socket.SOCK_STREAM
                    )[0][4][0]
                    print(f"Resolved {conn_params['host']} to IPv4: {ip_address}")
                    conn_params['host'] = ip_address
                except Exception as e:
                    print(f"IPv4 resolution failed, using hostname: {e}")
                
                conn = psycopg2.connect(**conn_params)
                print("✓ Database connection successful")
                return conn
                
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 5  # Progressive backoff
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print("All connection attempts failed")
                    # Auto-switch to REST API if PostgreSQL fails
                    if not self.use_rest_api:
                        print("🔄 Switching to REST API mode due to connection issues")
                        self.use_rest_api = True
                        return None
                    raise
    
    def create_table_if_not_exists(self):
        """Create the GSC metrics table if it doesn't exist"""
        if self.use_rest_api:
            print("✓ Using REST API mode - table creation handled by Supabase")
            return
            
        try:
            conn = self.get_db_connection()
            if conn is None:
                return  # REST API mode activated
                
            cur = conn.cursor()
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS gsc_metrics (
                    date DATE PRIMARY KEY,
                    clicks INTEGER DEFAULT 0,
                    impressions INTEGER DEFAULT 0,
                    ctr DECIMAL(10, 6) DEFAULT 0,
                    position DECIMAL(10, 2) DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Create index for faster queries
                CREATE INDEX IF NOT EXISTS idx_gsc_metrics_date ON gsc_metrics(date);
            """)
            
            conn.commit()
            cur.close()
            conn.close()
            print("✓ Database table ready")
            
        except Exception as e:
            print(f"Database table creation error: {e}")
            raise
    
    def test_db_connection(self):
        """Test database connection with detailed diagnostics"""
        if self.use_rest_api:
            return self.test_rest_api_connection()
            
        try:
            print("=== Database Connection Test ===")
            print(f"DATABASE_URL format check...")
            
            parsed = urllib.parse.urlparse(DATABASE_URL)
            print(f"Host: {parsed.hostname}")
            print(f"Port: {parsed.port}")
            print(f"Database: {parsed.path.lstrip('/')}")
            print(f"User: {parsed.username}")
            
            conn = self.get_db_connection()
            if conn is None:
                return self.test_rest_api_connection()  # Switched to REST API
                
            cur = conn.cursor()
            
            # Test basic query
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            print(f"✓ Connected to: {version}")
            
            # Test table access
            cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'gsc_metrics';")
            table_exists = cur.fetchone()[0] > 0
            print(f"✓ GSC metrics table exists: {table_exists}")
            
            cur.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            print("🔄 Trying REST API connection...")
            self.use_rest_api = True
            return self.test_rest_api_connection()
    
    def test_rest_api_connection(self):
        """Test Supabase REST API connection"""
        try:
            print("=== Testing Supabase REST API Connection ===")
            
            if not SUPABASE_URL or not SUPABASE_ANON_KEY:
                print("✗ Missing SUPABASE_URL or SUPABASE_ANON_KEY environment variables")
                print("Please add these to your GitHub secrets:")
                print("SUPABASE_URL: https://your-project.supabase.co")
                print("SUPABASE_ANON_KEY: your-anon-key-from-supabase-dashboard")
                return False
            
            headers = {
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                "Content-Type": "application/json"
            }
            
            # Test connection with a simple query
            response = requests.get(
                f"{SUPABASE_URL}/rest/v1/gsc_metrics?select=count&limit=1",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                print("✓ REST API connection successful")
                return True
            else:
                print(f"✗ REST API connection failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ REST API connection failed: {e}")
            return False
    
    def get_existing_dates(self) -> List[str]:
        """Get list of dates that already exist in database"""
        if self.use_rest_api:
            return self.get_existing_dates_rest_api()
            
        try:
            conn = self.get_db_connection()
            if conn is None:
                return self.get_existing_dates_rest_api()  # Switched to REST API
                
            cur = conn.cursor()
            
            cur.execute("SELECT date FROM gsc_metrics ORDER BY date")
            existing_dates = [row[0].isoformat() for row in cur.fetchall()]
            
            cur.close()
            conn.close()
            
            return existing_dates
            
        except Exception as e:
            print(f"Error fetching existing dates: {e}")
            return []
    
    def get_existing_dates_rest_api(self) -> List[str]:
        """Get existing dates via REST API"""
        try:
            headers = {
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            }
            
            response = requests.get(
                f"{SUPABASE_URL}/rest/v1/gsc_metrics?select=date&order=date.asc",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return [item['date'] for item in data]
            else:
                print(f"Error fetching existing dates via API: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching existing dates via API: {e}")
            return []
    
    def insert_batch_data(self, data_rows: List[Dict], skip_existing: bool = True):
        """Insert multiple rows of data into database"""
        if self.use_rest_api:
            return self.insert_batch_data_rest_api(data_rows, skip_existing)
            
        try:
            conn = self.get_db_connection()
            if conn is None:
                return self.insert_batch_data_rest_api(data_rows, skip_existing)  # Switched to REST API
                
            cur = conn.cursor()
            
            existing_dates = set(self.get_existing_dates()) if skip_existing else set()
            inserted_count = 0
            skipped_count = 0
            
            for row in data_rows:
                row_date = row['keys'][0]  # Date is the first dimension
                
                if skip_existing and row_date in existing_dates:
                    skipped_count += 1
                    continue
                
                cur.execute("""
                    INSERT INTO gsc_metrics (date, clicks, impressions, ctr, position, updated_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (date) DO UPDATE SET
                        clicks = EXCLUDED.clicks,
                        impressions = EXCLUDED.impressions,
                        ctr = EXCLUDED.ctr,
                        position = EXCLUDED.position,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    row_date,
                    row.get('clicks', 0),
                    row.get('impressions', 0),
                    row.get('ctr', 0.0),
                    row.get('position', 0.0)
                ))
                inserted_count += 1
            
            conn.commit()
            cur.close()
            conn.close()
            
            print(f"✓ Inserted {inserted_count} new records, skipped {skipped_count} existing")
            
        except Exception as e:
            print(f"Database insertion error: {e}")
            raise
    
    def insert_batch_data_rest_api(self, data_rows: List[Dict], skip_existing: bool = True):
        """Insert data via Supabase REST API with proper upsert handling"""
        try:
            headers = {
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                "Content-Type": "application/json"
            }
            
            inserted_count = 0
            updated_count = 0
            skipped_count = 0
            
            # Get existing dates if we need to skip them
            existing_dates = set(self.get_existing_dates()) if skip_existing else set()
            
            # Process each record individually for better control
            for row in data_rows:
                row_date = row['keys'][0]  # Date is the first dimension
                
                if skip_existing and row_date in existing_dates:
                    skipped_count += 1
                    continue
                
                record_data = {
                    "date": row_date,
                    "clicks": row.get('clicks', 0),
                    "impressions": row.get('impressions', 0),
                    "ctr": row.get('ctr', 0.0),
                    "position": row.get('position', 0.0)
                }
                
                # Check if record exists
                check_response = requests.get(
                    f"{SUPABASE_URL}/rest/v1/gsc_metrics?date=eq.{row_date}&select=date",
                    headers=headers,
                    timeout=30
                )
                
                if check_response.status_code == 200:
                    existing_records = check_response.json()
                    
                    if existing_records:
                        # Record exists - update it
                        update_headers = headers.copy()
                        update_headers["Prefer"] = "return=minimal"
                        
                        update_response = requests.patch(
                            f"{SUPABASE_URL}/rest/v1/gsc_metrics?date=eq.{row_date}",
                            json={
                                "clicks": record_data["clicks"],
                                "impressions": record_data["impressions"],
                                "ctr": record_data["ctr"],
                                "position": record_data["position"]
                            },
                            headers=update_headers,
                            timeout=30
                        )
                        
                        if update_response.status_code in [200, 204]:
                            updated_count += 1
                        else:
                            print(f"Update failed for {row_date}: {update_response.status_code} - {update_response.text}")
                    else:
                        # Record doesn't exist - insert it
                        insert_headers = headers.copy()
                        insert_headers["Prefer"] = "return=minimal"
                        
                        insert_response = requests.post(
                            f"{SUPABASE_URL}/rest/v1/gsc_metrics",
                            json=record_data,
                            headers=insert_headers,
                            timeout=30
                        )
                        
                        if insert_response.status_code in [200, 201]:
                            inserted_count += 1
                        else:
                            print(f"Insert failed for {row_date}: {insert_response.status_code} - {insert_response.text}")
                else:
                    print(f"Error checking existing record for {row_date}: {check_response.status_code}")
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.1)
            
            print(f"✓ Inserted {inserted_count} new records, updated {updated_count} existing records, skipped {skipped_count} via API")
            
        except Exception as e:
            print(f"API insertion error: {e}")
            raise
    
    def fetch_historical_data(self, months_back: int = 16):
        """Fetch historical data for the specified number of months"""
        print(f"\n=== Fetching {months_back} months of historical data ===")
        
        # Calculate date range (GSC data has ~3 day delay)
        end_date = date.today() - timedelta(days=3)
        start_date = end_date - timedelta(days=30 * months_back)
        
        print(f"Date range: {start_date} to {end_date}")
        
        # Fetch data in chunks to avoid API limits
        chunk_size = 30  # days
        current_date = start_date
        all_data = []
        
        while current_date <= end_date:
            chunk_end = min(current_date + timedelta(days=chunk_size - 1), end_date)
            
            print(f"Fetching chunk: {current_date} to {chunk_end}")
            chunk_data = self.fetch_gsc_data(current_date.isoformat(), chunk_end.isoformat())
            
            if chunk_data:
                all_data.extend(chunk_data)
                print(f"  ✓ Got {len(chunk_data)} records")
            else:
                print(f"  - No data for this chunk")
            
            current_date = chunk_end + timedelta(days=1)
            time.sleep(1)  # Be nice to the API
        
        if all_data:
            print(f"\nTotal records fetched: {len(all_data)}")
            self.insert_batch_data(all_data, skip_existing=True)
        else:
            print("No historical data found")
    
    def fetch_recent_data(self, days_back: int = 7):
        """Fetch recent data (for daily updates)"""
        print(f"\n=== Fetching last {days_back} days of data ===")
        
        end_date = date.today() - timedelta(days=3)  # GSC has ~3 day delay
        start_date = end_date - timedelta(days=days_back - 1)
        
        print(f"Date range: {start_date} to {end_date}")
        
        data = self.fetch_gsc_data(start_date.isoformat(), end_date.isoformat())
        
        if data:
            print(f"Found {len(data)} records")
            # For daily updates, we want to update existing records, so skip_existing=False
            self.insert_batch_data(data, skip_existing=False)
        else:
            print("No recent data found")
    
    def get_data_summary(self):
        """Get summary of data in database"""
        if self.use_rest_api:
            return self.get_data_summary_rest_api()
            
        try:
            conn = self.get_db_connection()
            if conn is None:
                return self.get_data_summary_rest_api()  # Switched to REST API
                
            cur = conn.cursor()
            
            cur.execute("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    SUM(clicks) as total_clicks,
                    SUM(impressions) as total_impressions
                FROM gsc_metrics
            """)
            
            result = cur.fetchone()
            cur.close()
            conn.close()
            
            if result and result[0] > 0:
                print(f"\n=== Database Summary ===")
                print(f"Total records: {result[0]}")
                print(f"Date range: {result[1]} to {result[2]}")
                print(f"Total clicks: {result[3]:,}")
                print(f"Total impressions: {result[4]:,}")
            else:
                print("No data in database")
                
        except Exception as e:
            print(f"Error getting summary: {e}")
    
    def get_data_summary_rest_api(self):
        """Get data summary via REST API"""
        try:
            headers = {
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            }
            
            # Get count
            count_response = requests.get(
                f"{SUPABASE_URL}/rest/v1/gsc_metrics?select=count",
                headers=headers,
                timeout=30
            )
            
            # Get aggregated data
            agg_response = requests.get(
                f"{SUPABASE_URL}/rest/v1/gsc_metrics?select=date,clicks,impressions&order=date.asc",
                headers=headers,
                timeout=30
            )
            
            if count_response.status_code == 200 and agg_response.status_code == 200:
                data = agg_response.json()
                
                if data:
                    total_records = len(data)
                    earliest_date = data[0]['date']
                    latest_date = data[-1]['date']
                    total_clicks = sum(item['clicks'] for item in data)
                    total_impressions = sum(item['impressions'] for item in data)
                    
                    print(f"\n=== Database Summary (via REST API) ===")
                    print(f"Total records: {total_records}")
                    print(f"Date range: {earliest_date} to {latest_date}")
                    print(f"Total clicks: {total_clicks:,}")
                    print(f"Total impressions: {total_impressions:,}")
                else:
                    print("No data in database")
            else:
                print("Error getting summary via API")
                
        except Exception as e:
            print(f"Error getting summary via API: {e}")

def main():
    parser = argparse.ArgumentParser(description='GSC Data Fetcher')
    parser.add_argument('--mode', choices=['historical', 'daily', 'test', 'summary', 'test-db'], 
                       default='daily', help='Operation mode')
    parser.add_argument('--months', type=int, default=16, 
                       help='Number of months for historical fetch')
    parser.add_argument('--days', type=int, default=7, 
                       help='Number of days for daily fetch')
    
    args = parser.parse_args()
    
    try:
        fetcher = GSCDataFetcher()
        
        if args.mode == 'test':
            print("=== Testing GSC Connection ===")
            fetcher.test_gsc_connection()
            print("\n=== Testing Database Connection ===")
            fetcher.test_db_connection()
            
        elif args.mode == 'test-db':
            fetcher.test_db_connection()
            
        elif args.mode == 'historical':
            fetcher.test_db_connection()
            fetcher.create_table_if_not_exists()
            fetcher.fetch_historical_data(args.months)
            fetcher.get_data_summary()
            
        elif args.mode == 'daily':
            fetcher.test_db_connection()
            fetcher.create_table_if_not_exists()
            fetcher.fetch_recent_data(args.days)
            fetcher.get_data_summary()
            
        elif args.mode == 'summary':
            fetcher.get_data_summary()
    
    except Exception as e:
        print(f"Script failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()