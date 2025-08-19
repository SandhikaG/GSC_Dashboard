import os
import psycopg2
from dotenv import load_dotenv

# Load .env file (so DATABASE_URL is available)
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def clear_database():
    """Delete all rows and reset ID counter"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        # Clears the table AND resets auto-increment id back to 1
        cur.execute("TRUNCATE TABLE gsc_metrics RESTART IDENTITY;")

        conn.commit()
        cur.close()
        conn.close()
        print("✅ Table cleared & ID counter reset")
    except Exception as e:
        print(f"Database error while clearing: {e}")

if __name__ == "__main__":
    clear_database()
