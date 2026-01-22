import os
import pandas as pd
from datetime import datetime, timedelta
from supabase import create_client
from datetime import date


supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"]
)

end_date = datetime.utcnow().date() - timedelta(days=2)
start_date = end_date - timedelta(days=29)
all_dates = pd.date_range(start_date, end_date)
rows = []
offset = 0
page_size = 500

while True:
    resp = (
        supabase
        .table("gsc_metrics")
        .select("*")
        .gte("date", start_date.isoformat())
        .lte("date", end_date.isoformat())
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
file_path = "Data/gsc_last_30_days.csv"
df.to_csv(file_path, index=False)

print(f"✅ Exported {len(df):,} rows")
# Upload CSVs grouped by date (SAFE)
for d in all_dates:
    date_value = d.date()
    group = df[df["date"].dt.date == date_value]

    if group.empty:
        print(f"⚠️ No rows for {date_value}, uploading empty CSV")
    
    file_name = f"gsc_{date_value}.csv"
    file_path = f"Data/{file_name}"

    group.to_csv(file_path, index=False)

    with open(file_path, "rb") as f:
        supabase.storage.from_("gsc-exports").upload(
            path=file_name,
            file=f,
            file_options={
                "content-type": "text/csv",
                "upsert": "true"
            },
        )

    print(f"✅ Uploaded {file_name} ({len(group):,} rows)")
