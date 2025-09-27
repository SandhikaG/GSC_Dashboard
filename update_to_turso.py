import pandas as pd
import asyncio
from libsql_client import create_client

async def insert_row(client, query, row_data):
    """Insert a single row"""
    try:
        await client.execute(query, row_data)
        return True
    except Exception as e:
        print(f"❌ Error inserting row: {e}")
        return False

async def main(): 
    # Connect to your Turso DB using HTTPS
    client = create_client(
        url="https://gsc-metrics-archana.aws-ap-south-1.turso.io",
        auth_token="eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJhIjoicnciLCJpYXQiOjE3NTc0OTM4ODIsImlkIjoiYmJmMjk1YjMtYTk2YS00NjBjLWE5YWYtMDlhMWVjZGViMDQzIiwicmlkIjoiNDQ1ZjI3NTAtMjk5YS00MGI5LWIwYzAtZjhmYjJjYmRhMDk3In0.DwkzaRyDqISqGmRaFj6qgpFUf-Fl2hXMbKsSWGUZ1kiYQ47W99aIRTIANsm-BZhOOaIJxsh2xkWDWd5VeZr2AQ"
    )

    try:
        # ✅ Ensure table exists
        await client.execute("""
        CREATE TABLE IF NOT EXISTS gsc_data (
            date TEXT,
            country TEXT,
            query TEXT,
            page TEXT,
            clicks INTEGER,
            impressions INTEGER,
            ctr REAL,
            position REAL,
            PRIMARY KEY (date, country, query, page)
        );
        """)
        print("✅ Table ready")

        # ✅ Load CSV with COMMA separator
        df = pd.read_csv(r"C:\Users\Naveen\Downloads\gsc_data_day_by_day (2).csv", sep=",")
        
        # ✅ RESUME FROM WHERE YOU LEFT OFF
        start_row = 3390000  # Last processed row was 3,390,000
        if start_row < len(df):
            df = df.iloc[start_row:]
            print(f"✅ Resuming from row {start_row + 1} (processing {len(df)} remaining rows)")
        else:
            print("✅ All rows have already been processed!")
            return

        # Debug: Print column names and first few rows
        print("CSV Columns:", df.columns.tolist())
        print("Number of rows to process:", len(df))
        print("First few rows:")
        print(df.head())

        # ✅ Prepare upsert query
        upsert_query = """
            INSERT INTO gsc_data (date, country, query, page, clicks, impressions, ctr, position)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, country, query, page)
            DO UPDATE SET
                clicks = excluded.clicks,
                impressions = excluded.impressions,
                ctr = excluded.ctr,
                position = excluded.position
        """

        # ✅ Process in concurrent batches
        batch_size = 5000  # Adjust based on your system performance
        semaphore = asyncio.Semaphore(10)  # Limit concurrent connections

        async def insert_with_semaphore(row_data):
            async with semaphore:
                return await insert_row(client, upsert_query, row_data)

        total_rows = len(df)
        successful_inserts = 0

        # Process in batches
        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:i + batch_size]
            
            # Create tasks for all rows in batch
            tasks = []
            for _, row in batch.iterrows():
                row_data = [
                    str(row["date"]),
                    str(row["country"]),
                    str(row["query"]),
                    str(row["page"]),
                    int(row["clicks"]),
                    int(row["impressions"]),
                    float(row["ctr"]),
                    float(row["position"])
                ]
                tasks.append(insert_with_semaphore(row_data))
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_successes = sum(1 for r in results if r is True)
            successful_inserts += batch_successes
            
            # Show progress with original row numbers
            original_start_row = start_row + i
            original_end_row = min(start_row + i + batch_size, start_row + total_rows)
            print(f"✅ Processed batch {original_start_row//batch_size + 1}: {original_end_row}/{start_row + total_rows} rows (Success: {batch_successes}/{len(tasks)})")

        print(f"✅ Successfully inserted/updated {successful_inserts} rows (resumed from row {start_row})")

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())