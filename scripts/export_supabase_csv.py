import os
import pandas as pd
from datetime import datetime, timedelta
from supabase import create_client

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)

end_date = datetime.utcnow().date() - timedelta(days=2)
start_date = end_date - timedelta(days=29)

rows = []
offset = 0
page_size = 1000

while True:
    resp = (
        supabase
        .table("gsc_metrics")
        .select("*")
        .gte("date", start_date.isoformat())
        .lte("date", end_date.isoformat())
        .order("date")
        .range(offset, offset + page_size - 1)
        .execute()
    )

    if not resp.data:
        break

    rows.extend(resp.data)
    if len(resp.data) < page_size:
        break

    offset += page_size

df = pd.DataFrame(rows)
df['date'] = pd.to_datetime(df['date'])

os.makedirs("Data", exist_ok=True)
df.to_csv("Data/gsc_last_30_days.csv", index=False)

print(f"✅ Exported {len(df):,} rows")
