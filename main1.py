from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import io
import warnings
from pydantic import BaseModel
import json

warnings.filterwarnings('ignore')

app = FastAPI(
    title="GSC Analytics API",
    description="FastAPI server for Google Search Console data analysis",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store processed data (in production, use a proper database or cache)
processed_data = {}

# Pydantic models for request/response
class FilterParams(BaseModel):
    country: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class MetricsResponse(BaseModel):
    total_clicks: int
    total_impressions: int
    avg_ctr: float
    avg_position: float
    unique_queries: int
    date_range: Dict[str, str]

def load_and_preprocess_data(file_content: bytes) -> pd.DataFrame:
    """Load and preprocess GSC data"""
    try:
        # Read CSV from bytes
        df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        
        # Try different date formats automatically
        try:
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        except ValueError:
            try:
                df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
            except ValueError:
                df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
        
        df['ctr'] = df['ctr'].astype(float)
        df['position'] = df['position'].astype(float)
        df['clicks'] = df['clicks'].astype(int)
        df['impressions'] = df['impressions'].astype(int)
        
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV file: {str(e)}")

def apply_filters(df: pd.DataFrame, filters: FilterParams) -> pd.DataFrame:
    """Apply filters to the dataframe"""
    filtered_df = df.copy()
    
    if filters.country and filters.country.lower() != 'all':
        filtered_df = filtered_df[filtered_df['country'] == filters.country]
    
    if filters.start_date:
        start_date = pd.to_datetime(filters.start_date).date()
        filtered_df = filtered_df[filtered_df['date'].dt.date >= start_date]
    
    if filters.end_date:
        end_date = pd.to_datetime(filters.end_date).date()
        filtered_df = filtered_df[filtered_df['date'].dt.date <= end_date]
    
    return filtered_df

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process GSC CSV file"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        content = await file.read()
        df = load_and_preprocess_data(content)
        
        # Store processed data (in production, use proper storage)
        processed_data['main_df'] = df
        
        return {
            "message": "File uploaded successfully",
            "rows": len(df),
            "date_range": {
                "start": df['date'].min().strftime('%Y-%m-%d'),
                "end": df['date'].max().strftime('%Y-%m-%d')
            },
            "countries": sorted(df['country'].unique().tolist()),
            "columns": df.columns.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/metrics")
async def get_metrics(filters: FilterParams = None):
    """Get key performance metrics"""
    if 'main_df' not in processed_data:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
    
    df = processed_data['main_df']
    
    if filters:
        df = apply_filters(df, filters)
    
    total_clicks = int(df['clicks'].sum())
    total_impressions = int(df['impressions'].sum())
    avg_ctr = float(total_clicks / total_impressions if total_impressions > 0 else 0)
    avg_position = float(df['position'].mean())
    unique_queries = int(df['query'].nunique())
    
    return {
        "total_clicks": total_clicks,
        "total_impressions": total_impressions,
        "avg_ctr": avg_ctr,
        "avg_position": avg_position,
        "unique_queries": unique_queries,
        "date_range": {
            "start": df['date'].min().strftime('%Y-%m-%d'),
            "end": df['date'].max().strftime('%Y-%m-%d')
        }
    }

@app.post("/opportunity-analysis")
async def get_opportunity_analysis(filters: FilterParams = None):
    """Get keyword opportunity quadrant analysis"""
    if 'main_df' not in processed_data:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
    
    df = processed_data['main_df']
    
    if filters:
        df = apply_filters(df, filters)
    
    # Create aggregated data
    df_agg = df.groupby('query').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'position': 'mean',
        'ctr': 'mean'
    }).reset_index()
    
    df_agg['ctr_calculated'] = df_agg['clicks'] / df_agg['impressions']
    df_agg = df_agg[df_agg['impressions'] > 50]  # Filter low impression queries
    
    median_impressions = float(df_agg['impressions'].median())
    median_ctr = float(df_agg['ctr_calculated'].median())
    
    # Categorize queries
    high_opp = df_agg[
        (df_agg['impressions'] > median_impressions) &
        (df_agg['ctr_calculated'] < median_ctr)
    ].sort_values('impressions', ascending=False)
    
    top_performers = df_agg[
        (df_agg['impressions'] > median_impressions) &
        (df_agg['ctr_calculated'] > median_ctr)
    ].sort_values('clicks', ascending=False)
    
    high_intent = df_agg[
        (df_agg['impressions'] <= median_impressions) &
        (df_agg['ctr_calculated'] > median_ctr)
    ].sort_values('ctr_calculated', ascending=False)
    
    low_priority = df_agg[
        (df_agg['impressions'] <= median_impressions) &
        (df_agg['ctr_calculated'] <= median_ctr)
    ].sort_values('impressions', ascending=False)
    
    return {
        "scatter_data": df_agg.to_dict('records'),
        "median_impressions": median_impressions,
        "median_ctr": median_ctr,
        "categories": {
            "biggest_opportunities": high_opp.to_dict('records'),
            "top_performers": top_performers.to_dict('records'),
            "high_intent": high_intent.to_dict('records'),
            "low_priority": low_priority.to_dict('records')
        }
    }

@app.post("/trending-analysis")
async def get_trending_analysis(filters: FilterParams = None):
    """Get trending queries analysis"""
    if 'main_df' not in processed_data:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
    
    df = processed_data['main_df']
    
    if filters:
        df = apply_filters(df, filters)
    
    if df['date'].nunique() < 8:
        return {"message": "Need at least 8 days of data for trend analysis", "data": None}
    
    latest_date = df['date'].max()
    period2_start = latest_date - timedelta(days=4)
    period1_end = period2_start - timedelta(days=1)
    period1_start = period1_end - timedelta(days=3)
    
    period1_data = df[(df['date'] >= period1_start) & (df['date'] <= period1_end)]
    period2_data = df[(df['date'] >= period2_start) & (df['date'] <= latest_date)]
    
    period1_agg = period1_data.groupby('query').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'position': 'mean'
    }).rename(columns={'clicks': 'clicks_p1', 'impressions': 'impressions_p1', 'position': 'position_p1'})
    
    period2_agg = period2_data.groupby('query').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'position': 'mean'
    }).rename(columns={'clicks': 'clicks_p2', 'impressions': 'impressions_p2', 'position': 'position_p2'})
    
    trend_df = pd.merge(period1_agg, period2_agg, on='query', how='outer').fillna(0)
    
    # Calculate changes
    trend_df['click_change'] = trend_df['clicks_p2'] - trend_df['clicks_p1']
    trend_df['click_pct_change'] = (trend_df['click_change'] / trend_df['clicks_p1'].replace(0, np.nan)) * 100
    trend_df['position_change'] = trend_df['position_p2'] - trend_df['position_p1']
    
    # Create improvement score
    trend_df['improvement_score'] = (
        trend_df['click_pct_change'].fillna(0) - 
        (trend_df['position_change'] * 10)
    )
    
    # Filter for meaningful queries
    trend_df = trend_df[(trend_df['clicks_p1'] > 5) | (trend_df['clicks_p2'] > 5)]
    
    # Sort and get top/bottom
    trend_df = trend_df.sort_values('improvement_score', ascending=False)
    
    winners = trend_df.head(20).reset_index()
    losers = trend_df.tail(20).reset_index()
    
    return {
        "winners": winners.to_dict('records'),
        "losers": losers.to_dict('records'),
        "period_info": {
            "period1_start": period1_start.strftime('%Y-%m-%d'),
            "period1_end": period1_end.strftime('%Y-%m-%d'),
            "period2_start": period2_start.strftime('%Y-%m-%d'),
            "period2_end": latest_date.strftime('%Y-%m-%d')
        }
    }

@app.post("/country-analysis")
async def get_country_analysis(filters: FilterParams = None):
    """Get country-wise performance analysis"""
    if 'main_df' not in processed_data:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
    
    df = processed_data['main_df']
    
    if filters:
        df = apply_filters(df, filters)
    
    country_perf = df.groupby('country').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'position': 'mean'
    }).reset_index()
    
    country_perf['ctr'] = country_perf['clicks'] / country_perf['impressions']
    country_perf = country_perf.sort_values('clicks', ascending=False)
    
    return {
        "country_performance": country_perf.to_dict('records'),
        "top_countries": country_perf.head(15).to_dict('records')
    }

@app.post("/content-gaps")
async def get_content_gaps(filters: FilterParams = None):
    """Get content gap opportunities"""
    if 'main_df' not in processed_data:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
    
    df = processed_data['main_df']
    
    if filters:
        df = apply_filters(df, filters)
    
    content_gaps = df.groupby('query').agg({
        'impressions': 'sum',
        'clicks': 'sum',
        'position': 'mean'
    }).reset_index()
    
    content_gaps['ctr'] = content_gaps['clicks'] / content_gaps['impressions']
    
    # Filter for high impression, low performing queries
    gaps = content_gaps[
        (content_gaps['impressions'] > content_gaps['impressions'].quantile(0.7)) &
        (content_gaps['ctr'] < content_gaps['ctr'].quantile(0.3)) &
        (content_gaps['position'] > 3)
    ].sort_values('impressions', ascending=False)
    
    gaps['opportunity_score'] = gaps['impressions'] * (1 - gaps['ctr'])
    
    return {
        "content_gaps": gaps.head(20).to_dict('records'),
        "total_gaps": len(gaps)
    }

@app.post("/page-performance")
async def get_page_performance(filters: FilterParams = None):
    """Get page-level performance analysis"""
    if 'main_df' not in processed_data:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
    
    df = processed_data['main_df']
    
    if filters:
        df = apply_filters(df, filters)
    
    page_perf = df.groupby('page').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'position': 'mean',
        'query': 'nunique'
    }).reset_index()
    
    page_perf['ctr'] = page_perf['clicks'] / page_perf['impressions']
    page_perf = page_perf.rename(columns={'query': 'unique_queries'})
    page_perf = page_perf.sort_values('clicks', ascending=False)
    
    return {
        "page_performance": page_perf.head(20).to_dict('records'),
        "total_pages": len(page_perf)
    }

@app.post("/time-series")
async def get_time_series(filters: FilterParams = None):
    """Get performance over time"""
    if 'main_df' not in processed_data:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
    
    df = processed_data['main_df']
    
    if filters:
        df = apply_filters(df, filters)
    
    daily_perf = df.groupby('date').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'position': 'mean'
    }).reset_index()
    
    daily_perf['ctr'] = daily_perf['clicks'] / daily_perf['impressions']
    daily_perf['date'] = daily_perf['date'].dt.strftime('%Y-%m-%d')
    
    return {
        "time_series": daily_perf.to_dict('records')
    }

@app.get("/countries")
async def get_available_countries():
    """Get list of available countries in the dataset"""
    if 'main_df' not in processed_data:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
    
    df = processed_data['main_df']
    countries = ['All'] + sorted(df['country'].unique().tolist())
    
    return {"countries": countries}

@app.get("/date-range")
async def get_date_range():
    """Get available date range in the dataset"""
    if 'main_df' not in processed_data:
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload a CSV file first.")
    
    df = processed_data['main_df']
    
    return {
        "start_date": df['date'].min().strftime('%Y-%m-%d'),
        "end_date": df['date'].max().strftime('%Y-%m-%d'),
        "total_days": df['date'].nunique()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "GSC Analytics API is running"}

@app.get("/data-status")
async def get_data_status():
    """Check if data is loaded"""
    if 'main_df' not in processed_data:
        return {"data_loaded": False, "message": "No data uploaded"}
    
    df = processed_data['main_df']
    return {
        "data_loaded": True,
        "rows": len(df),
        "date_range": {
            "start": df['date'].min().strftime('%Y-%m-%d'),
            "end": df['date'].max().strftime('%Y-%m-%d')
        },
        "countries": len(df['country'].unique()),
        "queries": df['query'].nunique()
    }

# Export endpoints
@app.post("/export/opportunities")
async def export_opportunities(filters: FilterParams = None):
    """Export opportunity analysis data as JSON"""
    opportunity_data = await get_opportunity_analysis(filters)
    return JSONResponse(content=opportunity_data)

@app.post("/export/content-gaps")
async def export_content_gaps(filters: FilterParams = None):
    """Export content gaps data as JSON"""
    content_gaps_data = await get_content_gaps(filters)
    return JSONResponse(content=content_gaps_data)

@app.post("/export/page-performance")
async def export_page_performance(filters: FilterParams = None):
    """Export page performance data as JSON"""
    page_perf_data = await get_page_performance(filters)
    return JSONResponse(content=page_perf_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)