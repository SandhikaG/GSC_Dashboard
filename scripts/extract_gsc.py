from google.oauth2 import service_account
from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional
import os
from supabase import create_client, Client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedGSCDataExtractor:
    def __init__(self, service_account_file: str, property_url: str):
        self.service_account_file = service_account_file
        self.property_url = property_url
        self.scopes = ["https://www.googleapis.com/auth/webmasters.readonly"]
        self.service = self._initialize_service()
        self.row_limit_per_request = 25000
    def _init_supabase(self) -> Client:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        return create_client(url, key)
    def push_to_supabase(self, df: pd.DataFrame, table: str = "gsc_metrics"):
        if df.empty:
            logger.warning("No data to push to Supabase")
            return
        df = df.drop_duplicates(subset=["date", "page", "query"])
        supabase = self._init_supabase()

        # Convert dataframe to records
        records = df.to_dict(orient="records")

        batch_size = 1000
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            response = supabase.table(table).upsert(batch).execute()

            if response.data is None:
                logger.error(f"Supabase insert failed at batch {i}")
            else:
                logger.info(f"Inserted batch {i} → {i + len(batch)}")

    def _extract_blog_category(self, url: str) -> str:
      """
      Extract blog category from page URL
      Example:
      /blog/security/what-is-x → security
      """
      try:
          parts = url.split("/")
          if "blog" in parts:
              blog_index = parts.index("blog")
              if blog_index + 1 < len(parts):
                  return parts[blog_index + 1]
      except Exception:
          pass

      return "other"
    def _initialize_service(self):
        """Initialize the Google Search Console service"""
        creds = service_account.Credentials.from_service_account_file(
            self.service_account_file, scopes=self.scopes
        )
        service = build("searchconsole", "v1", credentials=creds, cache_discovery=False)
        return service

    def _make_request(self, start_date: str, end_date: str, dimensions: List[str],
                     filters: List[Dict] = None, row_limit: int = 25000,
                     start_row: int = 0, search_type: str = 'web') -> Optional[Dict]:
        """Make a single API request to GSC"""
        request_body = {
            "startDate": start_date,
            "endDate": end_date,
            "dimensions": dimensions,
            "rowLimit": row_limit,
            "startRow": start_row,
            "searchType": search_type,
            "dataState": "final"
        }

        if dimensions:
            request_body["aggregationType"] = "auto"

        if filters:
            request_body["dimensionFilterGroups"] = [{
                "filters": filters
            }]

        try:
            response = self.service.searchanalytics().query(
                siteUrl=self.property_url,
                body=request_body
            ).execute()
            return response
        except Exception as e:
            logger.error(f"Error making request: {e}")
            return None

    def _format_response_to_df(self, response: Dict, dimensions: List[str]) -> Optional[pd.DataFrame]:
        """Convert API response to pandas DataFrame"""
        if not response or 'rows' not in response:
            return None

        rows_data = []
        for row in response['rows']:
            row_dict = {}

            if 'keys' in row and dimensions:
                for i, key in enumerate(row['keys']):
                    if i < len(dimensions):
                        row_dict[dimensions[i]] = key

            row_dict.update({
                'clicks': row.get('clicks', 0),
                'impressions': row.get('impressions', 0),
                'ctr': row.get('ctr', 0.0),
                'position': row.get('position', 0.0)
            })
            rows_data.append(row_dict)

        return pd.DataFrame(rows_data)

    def get_single_day_data(self, date: str, dimensions: List[str],
                           max_rows: int = 200000) -> Optional[pd.DataFrame]:
        """
        Get data for a single day with comprehensive pagination and segmentation
        """
        logger.info(f"Extracting data for {date}")

        # First, try direct extraction
        data = self._get_data_with_pagination(date, date, dimensions, max_rows)

        if data is not None and len(data) > 0:
            logger.info(f"Direct extraction for {date}: {len(data)} rows")
            return data

        # If direct extraction fails or hits limits, try country segmentation
        logger.info(f"Trying country segmentation for {date}")
        data = self._get_data_by_country_segments(date, date, dimensions)

        if data is not None and len(data) > 0:
            logger.info(f"Country segmentation for {date}: {len(data)} rows")
            return data

        # If still no data, try query segmentation (for very high volume days)
        logger.info(f"Trying query segmentation for {date}")
        data = self._get_data_by_query_segments(date, date, dimensions)
        
        if data is not None and len(data) > 0:
            logger.info(f"Query segmentation for {date}: {len(data)} rows")
            return data
       
        logger.warning(f"No data extracted for {date}")
        return None

    def _get_data_with_pagination(self, start_date: str, end_date: str,
                                 dimensions: List[str], max_rows: int) -> Optional[pd.DataFrame]:
        """Get data with pagination"""
        all_dataframes = []
        start_row = 0
        total_fetched = 0

        while total_fetched < max_rows:
            remaining_rows = min(self.row_limit_per_request, max_rows - total_fetched)

            response = self._make_request(
                start_date, end_date, dimensions,
                row_limit=remaining_rows,
                start_row=start_row
            )

            if not response or 'rows' not in response:
                break

            df_chunk = self._format_response_to_df(response, dimensions)
            if df_chunk is None or df_chunk.empty:
                break

            all_dataframes.append(df_chunk)
            fetched_rows = len(df_chunk)
            total_fetched += fetched_rows
            start_row += fetched_rows

            if fetched_rows < remaining_rows:
                break

            time.sleep(0.1)  # Rate limiting

        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            return combined_df.drop_duplicates()
        return None

    def _get_data_by_country_segments(self, start_date: str, end_date: str,
                                     target_dimensions: List[str]) -> Optional[pd.DataFrame]:
        """Get data by segmenting by country"""
        # Get list of countries for this date
        countries_response = self._make_request(
            start_date, end_date,
            dimensions=['country'],
            row_limit=1000
        )

        if not countries_response or 'rows' not in countries_response:
            return None

        countries = [row['keys'][0] for row in countries_response['rows']]
        all_dataframes = []

        for country in countries:
            filters = [{
                "dimension": "country",
                "operator": "equals",
                "expression": country
            }]

            query_dimensions = target_dimensions.copy()
            if 'country' not in query_dimensions:
                query_dimensions.append('country')

            country_data = self._get_filtered_data(
                start_date, end_date, query_dimensions, filters, max_rows=50000
            )

            if country_data is not None and not country_data.empty:
                all_dataframes.append(country_data)

            time.sleep(0.1)

        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            return combined_df.drop_duplicates()
        return None

    def _get_data_by_query_segments(self, start_date: str, end_date: str,
                                   target_dimensions: List[str]) -> Optional[pd.DataFrame]:
        """Get data by segmenting by top queries (for very high volume days)"""
        # Get top 100 queries for this date
        queries_response = self._make_request(
            start_date, end_date,
            dimensions=['query'],
            row_limit=100
        )

        if not queries_response or 'rows' not in queries_response:
            return None

        queries = [row['keys'][0] for row in queries_response['rows']]
        all_dataframes = []

        for query in queries:
            filters = [{
                "dimension": "query",
                "operator": "equals",
                "expression": query
            }]

            query_dimensions = target_dimensions.copy()
            if 'query' not in query_dimensions:
                query_dimensions.append('query')

            query_data = self._get_filtered_data(
                start_date, end_date, query_dimensions, filters, max_rows=25000
            )

            if query_data is not None and not query_data.empty:
                all_dataframes.append(query_data)

            time.sleep(0.1)

        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            return combined_df.drop_duplicates()
        return None

    def _get_filtered_data(self, start_date: str, end_date: str,
                          dimensions: List[str], filters: List[Dict],
                          max_rows: int = 50000) -> Optional[pd.DataFrame]:
        """Get data with specific filters"""
        all_dataframes = []
        start_row = 0

        while start_row < max_rows:
            remaining_rows = min(self.row_limit_per_request, max_rows - start_row)

            response = self._make_request(
                start_date, end_date, dimensions,
                filters=filters,
                row_limit=remaining_rows,
                start_row=start_row
            )

            if not response or 'rows' not in response:
                break

            df_chunk = self._format_response_to_df(response, dimensions)
            if df_chunk is None or df_chunk.empty:
                break

            all_dataframes.append(df_chunk)
            fetched_rows = len(df_chunk)
            start_row += fetched_rows

            if fetched_rows < remaining_rows:
                break

            time.sleep(0.1)

        if all_dataframes:
            return pd.concat(all_dataframes, ignore_index=True)
        return None

    def _ensure_column_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame has the correct column order and all required columns"""
        required_columns = ['date', 'country', 'query', 'page', 'blog_category','clicks', 'impressions', 'ctr', 'position']

        for col in required_columns:
            if col not in df.columns:
                if col in ['clicks', 'impressions']:
                    df[col] = 0
                elif col in ['ctr', 'position']:
                    df[col] = 0.0
                else:
                    df[col] = 'Unknown'

        df = df[required_columns]

        # Ensure proper data types
        df['clicks'] = df['clicks'].astype(int)
        df['impressions'] = df['impressions'].astype(int)
        df['ctr'] = df['ctr'].astype(float)
        df['position'] = df['position'].astype(float)

        return df

    def extract_data_day_by_day(self, start_date: str, end_date: str,
                               export_filename: str) -> pd.DataFrame:
        export_dir = os.path.dirname(export_filename)
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
        target_dimensions = ['date', 'country', 'query', 'page']

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        date_list = []
        current_date = start_dt
        while current_date <= end_dt:
            date_list.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)

        total_rows = 0
        total_clicks = 0
        total_impressions = 0

        if os.path.exists(export_filename):
            os.remove(export_filename)
            logger.info(f"Removed existing file: {export_filename}")

        for i, date in enumerate(date_list, 1):
            logger.info(f"Processing date {i}/{len(date_list)}: {date}")

            daily_data = self.get_single_day_data(date, target_dimensions)

            if daily_data is not None and not daily_data.empty:
                daily_data["blog_category"] = daily_data["page"].apply(
                    self._extract_blog_category
                )
                daily_data = self._ensure_column_order(daily_data)

                if not os.path.exists(export_filename):
                    # First write - include header
                    daily_data.to_csv(export_filename, index=False, encoding='utf-8')
                    logger.info(f"Created new file: {export_filename}")
                else:
                    # Append without header
                    daily_data.to_csv(export_filename, mode='a', header=False,
                                    index=False, encoding='utf-8')

                day_rows = len(daily_data)
                day_clicks = daily_data['clicks'].sum()
                day_impressions = daily_data['impressions'].sum()

                total_rows += day_rows
                total_clicks += day_clicks
                total_impressions += day_impressions

                logger.info(f"✅ {date}: {day_rows:,} rows, {day_clicks:,} clicks, {day_impressions:,} impressions")

            else:
                logger.warning(f"❌ {date}: No data found")

            time.sleep(0.5)

        print("\n" + "="*60)
        print("📊 GSC DATA EXTRACTION COMPLETE")
        print("="*60)
        print(f"Date Range: {start_date} to {end_date}")
        print(f"Total Days Processed: {len(date_list)}")
        print(f"Total Rows Extracted: {total_rows:,}")
        print(f"Total Clicks: {total_clicks:,}")
        print(f"Total Impressions: {total_impressions:,}")
        print(f"💾 Data saved to: {export_filename}")

        if os.path.exists(export_filename):
            final_data = pd.read_csv(export_filename)
            print(f"📋 Final dataset shape: {final_data.shape}")
            print(f"📋 Sample of final data:")
            print(final_data.head())
            self.push_to_supabase(final_data)
            return final_data
        else:
            return pd.DataFrame()

def extract_gsc_data_enhanced(start_date: str, end_date: str,
                            service_account_file: str, property_url: str,
                            export_filename: str = 'gsc_data_enhanced.csv') -> pd.DataFrame:
    extractor = EnhancedGSCDataExtractor(service_account_file, property_url)

    logger.info(f"Starting enhanced GSC data extraction")
    logger.info(f"Date Range: {start_date} to {end_date}")
    logger.info(f"Output File: {export_filename}")

    data = extractor.extract_data_day_by_day(start_date, end_date, export_filename)

    return data
if __name__ == "__main__":
    # Calculate previous month date range
    end_date_dt = datetime.today() - timedelta(days=2)
    start_date_dt = end_date_dt - timedelta(days=29)

    start_date = start_date_dt.strftime("%Y-%m-%d")
    end_date = end_date_dt.strftime("%Y-%m-%d")

    SERVICE_ACCOUNT_FILE = "gsc_key.json"  # GitHub workflow creates this file
    PROPERTY_URL = "https://www.fortinet.com"

    data = extract_gsc_data_enhanced(
        start_date=start_date,
        end_date=end_date,
        service_account_file=SERVICE_ACCOUNT_FILE,
        property_url=PROPERTY_URL,
        export_filename='Data/gsc_data_day_by_day.csv'
    )

    print(f"\n✅ Extracted data for {start_date} to {end_date}")