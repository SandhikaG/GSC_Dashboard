import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import altair as alt
import warnings
warnings.filterwarnings('ignore')
import os
from supabase import create_client, Client
from dotenv import load_dotenv


load_dotenv()
st.set_page_config(
    page_title="GSC Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 1rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .insight-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3f4f6 100%);
        border-left: 5px solid #2196f3;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .section-header {
        color: #34495e;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        color: #7f8c8d;
        font-size: 0.9rem;
    }
    .recommendation-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .branded-section {
        background: linear-gradient(135deg, #e8f5e8 0%, #f0f8f0 100%);
        border-left: 4px solid #27ae60;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
    }
    .non-branded-section {
        background: linear-gradient(135deg, #fff3e0 0%, #fef7f0 100%);
        border-left: 4px solid #f39c12;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)
@st.cache_data(ttl=3600)
def load_data_from_supabase():
    supabase: Client = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_ANON_KEY"]
    )

    response = (
        supabase
        .table("gsc_metrics")
        .select("*")
        .order("date", desc=False)
        .execute()
    )

    if not response.data:
        return pd.DataFrame()

    df = pd.DataFrame(response.data)

    # Type safety (VERY IMPORTANT)
    df['date'] = pd.to_datetime(df['date'])
    df['clicks'] = df['clicks'].astype(int)
    df['impressions'] = df['impressions'].astype(int)
    df['ctr'] = df['ctr'].astype(float)
    df['position'] = df['position'].astype(float)

    return df


def is_branded_query(query, brand_terms=['forti']):
    """Check if a query contains branded terms"""
    query_lower = query.lower()
    return any(term in query_lower for term in brand_terms)

def get_spanish_speaking_countries():
    """Return list of Spanish-speaking countries (lowercase codes)"""
    return ['esp', 'mex', 'arg', 'col', 'chl', 'per', 'ven', 'ecu', 
            'gtm', 'cub', 'bol', 'dom', 'hnd', 'pry', 'slv', 'nic', 
            'cri', 'pan', 'ury', 'gnq', 'pri']

def create_spanish_market_analysis(df):
    """Create comprehensive Spanish-speaking markets analysis"""
    spanish_countries = get_spanish_speaking_countries()
    
    # Separate Spanish-speaking and rest of world (compare lowercase to be robust)
    df_spanish = df[df['country'].str.lower().isin(spanish_countries)].copy()
    df_row = df[~df['country'].str.lower().isin(spanish_countries)].copy()
    
    if len(df_spanish) == 0:
        return None, None, None, None, None
    
    # Aggregate metrics
    spanish_metrics = {
        'clicks': df_spanish['clicks'].sum(),
        'impressions': df_spanish['impressions'].sum(),
        'avg_ctr': df_spanish['clicks'].sum() / df_spanish['impressions'].sum() if df_spanish['impressions'].sum() > 0 else 0,
        'avg_position': df_spanish['position'].mean(),
        'unique_queries': df_spanish['query'].nunique(),
        'unique_pages': df_spanish['page'].nunique()
    }
    
    row_metrics = {
        'clicks': df_row['clicks'].sum(),
        'impressions': df_row['impressions'].sum(),
        'avg_ctr': df_row['clicks'].sum() / df_row['impressions'].sum() if df_row['impressions'].sum() > 0 else 0,
        'avg_position': df_row['position'].mean(),
        'unique_queries': df_row['query'].nunique(),
        'unique_pages': df_row['page'].nunique()
    }
    
    # Top queries in Spanish markets
    spanish_top_queries = df_spanish.groupby('query').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'position': 'mean'
    }).reset_index()
    spanish_top_queries['ctr'] = spanish_top_queries['clicks'] / spanish_top_queries['impressions']
    spanish_top_queries = spanish_top_queries.sort_values('clicks', ascending=False).head(30)
    
    # Top pages in Spanish markets
    spanish_top_pages = df_spanish.groupby('page').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'position': 'mean',
        'query': 'nunique'
    }).reset_index()
    spanish_top_pages['ctr'] = spanish_top_pages['clicks'] / spanish_top_pages['impressions']
    spanish_top_pages = spanish_top_pages.rename(columns={'query': 'unique_queries'})
    spanish_top_pages = spanish_top_pages.sort_values('clicks', ascending=False).head(30)
    
    # Country breakdown
    spanish_by_country = df_spanish.groupby('country').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'position': 'mean'
    }).reset_index()
    spanish_by_country['ctr'] = spanish_by_country['clicks'] / spanish_by_country['impressions']
    spanish_by_country = spanish_by_country.sort_values('clicks', ascending=False)
    
    return spanish_metrics, row_metrics, spanish_top_queries, spanish_top_pages, spanish_by_country

def create_opportunity_quadrant(df):
    df_agg = df.groupby('query').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'position': 'mean',
        'ctr': 'mean'
    }).reset_index()
    
    df_agg['ctr_calculated'] = df_agg['clicks'] / df_agg['impressions']

    df_agg = df_agg[df_agg['impressions'] > 50]
    
    if len(df_agg) == 0:
        return None, pd.DataFrame()
    
    median_impressions = df_agg['impressions'].median()
    median_ctr = df_agg['ctr_calculated'].median()
    
    fig = px.scatter(
        df_agg,
        x='impressions',
        y='ctr_calculated',
        size='clicks',
        hover_name='query',
        hover_data=['position', 'clicks', 'impressions'],
        title='<b>Keyword Opportunity Analysis</b>',
        labels={
            'impressions': 'Impressions',
            'ctr_calculated': 'Click-Through Rate (CTR)',
            'position': 'Avg Position'
        },
        log_x=True,
        color='position',
        color_continuous_scale='RdYlBu_r'
    )
    
    # Add quadrant lines
    fig.add_hline(y=median_ctr, line_dash="dash", line_color="red", opacity=0.7)
    fig.add_vline(x=median_impressions, line_dash="dash", line_color="red", opacity=0.7)
    
    # FIXED: Properly positioned quadrant labels
    fig.update_layout(
        annotations=[
            # Top Left Quadrant: High Intent, Low Volume (Low impressions, High CTR)
            dict(x=0.15, y=0.85, xref='paper', yref='paper',
                 text="<b>High Intent<br>Low Volume</b>", showarrow=False, 
                 font=dict(color="blue", size=12), bgcolor="rgba(255,255,255,0.8)",
                 bordercolor="blue", borderwidth=1),
            # Top Right Quadrant: Top Performers (High impressions, High CTR)
            dict(x=0.85, y=0.85, xref='paper', yref='paper',
                 text="<b>Top Performers</b>", showarrow=False, 
                 font=dict(color="green", size=12), bgcolor="rgba(255,255,255,0.8)",
                 bordercolor="green", borderwidth=1),
            # Bottom Left Quadrant: Low Priority (Low impressions, Low CTR)
            dict(x=0.15, y=0.15, xref='paper', yref='paper',
                 text="<b>Low Priority</b>", showarrow=False,
                 font=dict(color="gray", size=12), bgcolor="rgba(255,255,255,0.8)",
                 bordercolor="gray", borderwidth=1),
            # Bottom Right Quadrant: Biggest Opportunities (High impressions, Low CTR)
            dict(x=0.85, y=0.15, xref='paper', yref='paper',
                text="<b>Biggest<br>Opportunities</b>", showarrow=False, 
                font=dict(color="orange", size=12), bgcolor="rgba(255,255,255,0.8)",
                bordercolor="orange", borderwidth=1),
        ],
        height=600
    )
    
    return fig, df_agg

def create_branded_non_branded_analysis(opp_data):
    """Create separate analysis for branded vs non-branded queries"""
    if len(opp_data) == 0:
        return None, None, None, None
    
    # Calculate medians from the filtered data
    median_impressions = opp_data['impressions'].median()
    median_ctr = opp_data['ctr_calculated'].median()
    
    # Filter out all queries containing "https" in any form
    opp_data_filtered = opp_data[~opp_data['query'].str.contains('https', case=False, na=False)]
    
    # Separate branded and non-branded queries
    branded_data = opp_data_filtered[opp_data_filtered['query'].apply(is_branded_query)].reset_index(drop=True)
    non_branded_data = opp_data_filtered[~opp_data_filtered['query'].apply(is_branded_query)]
    
    # Branded opportunities and top performers
    branded_opp = branded_data[
        (branded_data['impressions'] > median_impressions) &
        (branded_data['ctr_calculated'] < median_ctr)
    ].sort_values('impressions', ascending=False)
    
    branded_top = branded_data[
        (branded_data['impressions'] > median_impressions) &
        (branded_data['ctr_calculated'] > median_ctr)
    ].sort_values('clicks', ascending=False)
    
    # Non-branded opportunities and top performers
    non_branded_opp = non_branded_data[
        (non_branded_data['impressions'] > median_impressions) &
        (non_branded_data['ctr_calculated'] < median_ctr)
    ].sort_values('impressions', ascending=False)
    
    non_branded_top = non_branded_data[
        (non_branded_data['impressions'] > median_impressions) &
        (non_branded_data['ctr_calculated'] > median_ctr)
    ].sort_values('clicks', ascending=False)
    
    return branded_opp, branded_top, non_branded_opp, non_branded_top

def create_trending_analysis(df):
    """Create trending queries analysis with FIXED position logic"""
    if df['date'].nunique() < 8:
        return None, None
    
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
    
    # FIXED: Position change calculation (lower position number = better ranking)
    trend_df['position_change'] = trend_df['position_p2'] - trend_df['position_p1']
    # Negative position change = improvement (moved up in rankings)
    # Positive position change = decline (moved down in rankings)
    
    # Create a combined improvement score
    # Higher clicks AND better position (lower number) = better
    trend_df['improvement_score'] = (
        trend_df['click_pct_change'].fillna(0) - 
        (trend_df['position_change'] * 10)  # Multiply by 10 to weight position changes
    )
    
    # Filter for meaningful queries
    trend_df = trend_df[(trend_df['clicks_p1'] > 5) | (trend_df['clicks_p2'] > 5)]
    
    # Sort by improvement score
    trend_df = trend_df.sort_values('improvement_score', ascending=False)
    
    return trend_df.head(20), trend_df.tail(20)

def create_country_blog_category_analysis(df):
    """
    Country → Blog Category performance aggregation
    """
    agg_df = (
        df.groupby(['country', 'blog_category'])
          .agg(
              clicks=('clicks', 'sum'),
              impressions=('impressions', 'sum'),
              avg_position=('position', 'mean')
          )
          .reset_index()
    )

    # Proper CTR calculation
    agg_df['ctr'] = agg_df['clicks'] / agg_df['impressions']
    agg_df = agg_df.sort_values(['country', 'clicks'], ascending=[True, False])

    return agg_df


def create_country_analysis(df):
    """Create country-wise performance analysis"""
    country_perf = df.groupby('country').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'position': 'mean'
    }).reset_index()
    
    country_perf['ctr'] = country_perf['clicks'] / country_perf['impressions']
    country_perf = country_perf.sort_values('clicks', ascending=False).head(15)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Clicks by Country', 'Impressions by Country', 
                    'CTR by Country', 'Average Position by Country'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Add traces
    fig.add_trace(
        go.Bar(x=country_perf['country'], y=country_perf['clicks'], 
            name='Clicks', marker_color='#3498db'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=country_perf['country'], y=country_perf['impressions'], 
            name='Impressions', marker_color='#2ecc71'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=country_perf['country'], y=country_perf['ctr'], 
            name='CTR', marker_color='#e74c3c'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=country_perf['country'], y=country_perf['position'], 
            name='Position', marker_color='#f39c12'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="<b>Geographic Performance Analysis</b>")
    fig.update_xaxes(tickangle=45)
    
    return fig, country_perf

def extract_content_gaps(df):
    """Identify content gap opportunities - FIXED to handle empty results"""
    # High impression, low CTR queries (content optimization opportunities)
    content_gaps = df.groupby('query').agg({
        'impressions': 'sum',
        'clicks': 'sum',
        'position': 'mean'
    }).reset_index()
    
    content_gaps['ctr'] = content_gaps['clicks'] / content_gaps['impressions']
    
    # Only proceed if we have data
    if len(content_gaps) == 0:
        return pd.DataFrame()
    
    # Filter for high impression, low performing queries
    impression_threshold = content_gaps['impressions'].quantile(0.7)
    ctr_threshold = content_gaps['ctr'].quantile(0.3)
    
    gaps = content_gaps[
        (content_gaps['impressions'] > impression_threshold) &
        (content_gaps['ctr'] < ctr_threshold) &
        (content_gaps['position'] > 3)
    ].sort_values('impressions', ascending=False)
    
    return gaps.head(20)

def create_page_performance_analysis(df):
    """Analyze page-level performance with filtering options"""
    page_perf = df.groupby('page').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'position': 'mean',
        'query': 'nunique'
    }).reset_index()
    
    page_perf['ctr'] = page_perf['clicks'] / page_perf['impressions']
    page_perf = page_perf.rename(columns={'query': 'unique_queries'})
    page_perf = page_perf.sort_values('clicks', ascending=False)
    
    # Create filtered version without product-downloads
    page_perf_filtered = page_perf[~page_perf['page'].str.contains('product-downloads', case=False, na=False)]
    
    # Create filtered version for data-sheets only
    page_perf_datasheets = page_perf[page_perf['page'].str.contains('/resources/data-sheets/', case=False, na=False)]
    
    return page_perf.head(50), page_perf_filtered.head(50), page_perf_datasheets.head(50)

def main():
    st.markdown('<h1 class="main-header">GSC Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Define the path to the data file
    #data_path = "Data/gsc_data_day_by_day.csv"

    try:
        # Load data from the fixed path
        with st.spinner("Loading data..."):
            df=load_data_from_supabase()
           # df = load_data(data_path)
        if df.empty:
            st.warning("No data available in Supabase yet.")
            st.stop()

        st.sidebar.header("Data Loaded")
        st.sidebar.success(f"Loaded {len(df):,} rows")
        
        # Date range info
        st.sidebar.info(f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
        # Sidebar filters
        st.sidebar.header("Filters")
        
        # Country filter
        countries = ['All'] + sorted(df['country'].unique().tolist())
        selected_country = st.sidebar.selectbox("Select Country", countries)
        
        # Date range filter
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(df['date'].min().date(), df['date'].max().date()),
            min_value=df['date'].min().date(),
            max_value=df['date'].max().date()
        )
        
        
        # Apply filters
        filtered_df = df.copy()
        if selected_country != 'All':
            filtered_df = filtered_df[filtered_df['country'] == selected_country]

        # 🔽 Country → Blog Category Aggregation
        country_blog_df = create_country_blog_category_analysis(filtered_df)

        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['date'].dt.date >= date_range[0]) &
                (filtered_df['date'].dt.date <= date_range[1])
            ]
        
        # Main dashboard content
        # Key Metrics
        st.markdown('<h2 class="section-header">Key Performance Metrics</h2>', unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_clicks = filtered_df['clicks'].sum()
        total_impressions = filtered_df['impressions'].sum()
        avg_ctr = total_clicks / total_impressions if total_impressions > 0 else 0
        avg_position = filtered_df['position'].mean()
        unique_queries = filtered_df['query'].nunique()
        
        with col1:
            st.metric("Total Clicks", f"{total_clicks:,}")
        with col2:
            st.metric("Total Impressions", f"{total_impressions:,}")
        with col3:
            st.metric("Average CTR", f"{avg_ctr:.2%}")
        with col4:
            st.metric("Average Position", f"{avg_position:.1f}")
        with col5:
            st.metric("Unique Queries", f"{unique_queries:,}")
        
        # Opportunity Quadrant
        st.markdown('<h2 class="section-header">Keyword Opportunity Analysis</h2>', unsafe_allow_html=True)
        
        with st.spinner("Creating opportunity analysis..."):
            opp_result = create_opportunity_quadrant(filtered_df)
            
            if opp_result[0] is not None:
                opp_fig, opp_data = opp_result
                st.plotly_chart(opp_fig, use_container_width=True)
                
                # NEW: Branded vs Non-branded analysis
                branded_opp, branded_top, non_branded_opp, non_branded_top = create_branded_non_branded_analysis(opp_data)
                
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.subheader("Key Insights - Branded vs Non-Branded Analysis")
                
                # Create tabs for better organization
                tab1, tab2 = st.tabs(["🎯 Biggest Opportunities", "🏆 Top Performers"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="branded-section">', unsafe_allow_html=True)
                        st.write(f"**Branded Opportunities (Forti):** {len(branded_opp)} queries")
                        if len(branded_opp) > 0:
                           st.dataframe(
                                branded_opp[['query', 'impressions', 'clicks', 'ctr_calculated', 'position']]
                                    .reset_index(drop=True)
                                    .style.format({
                                        'impressions': '{:,.0f}',
                                        'clicks': '{:,.0f}',
                                        'ctr_calculated': '{:.2%}',
                                        'position': '{:.1f}'
                                        }),
                                height=400,
                                use_container_width=True
                            )
                        else:
                            st.info("No branded opportunities found in current filter.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="non-branded-section">', unsafe_allow_html=True)
                        st.write(f"**Non-Branded Opportunities:** {len(non_branded_opp)} queries")
                        if len(non_branded_opp) > 0:
                            st.dataframe(
                                non_branded_opp[['query', 'impressions', 'clicks', 'ctr_calculated', 'position']]
                                .reset_index(drop=True)
                                .style.format({
                                    'impressions': '{:,.0f}',
                                    'clicks': '{:,.0f}',
                                    'ctr_calculated': '{:.2%}',
                                    'position': '{:.1f}'
                                }),
                                height=400,
                                use_container_width=True
                            )
                        else:
                            st.info("No non-branded opportunities found in current filter.")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                with tab2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="branded-section">', unsafe_allow_html=True)
                        st.write(f"**Branded Top Performers (Forti):** {len(branded_top)} queries")
                        if len(branded_top) > 0:
                            st.dataframe(
                                branded_top[['query', 'impressions', 'clicks', 'ctr_calculated', 'position']]
                                .reset_index(drop=True).style.format({
                                    'impressions': '{:,.0f}',
                                    'clicks': '{:,.0f}',
                                    'ctr_calculated': '{:.2%}',
                                    'position': '{:.1f}'
                                }),
                                height=400,
                                use_container_width=True
                            )
                        else:
                            st.info("No branded top performers found in current filter.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="non-branded-section">', unsafe_allow_html=True)
                        st.write(f"**Non-Branded Top Performers:** {len(non_branded_top)} queries")
                        if len(non_branded_top) > 0:
                            st.dataframe(
                                non_branded_top[['query', 'impressions', 'clicks', 'ctr_calculated', 'position']]
                                .reset_index(drop=True).style.format({
                                    'impressions': '{:,.0f}',
                                    'clicks': '{:,.0f}',
                                    'ctr_calculated': '{:.2%}',
                                    'position': '{:.1f}'
                                }),
                                height=400,
                                use_container_width=True
                            )
                        else:
                            st.info("No non-branded top performers found in current filter.")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No data available for opportunity analysis. Try adjusting your filters or ensure you have queries with more than 50 impressions.")
        
        # Country Analysis (if multiple countries)
        if filtered_df['country'].nunique() > 1:
            st.markdown('<h2 class="section-header">Geographic Performance Analysis</h2>', unsafe_allow_html=True)
            with st.spinner("Creating geographic analysis..."):
                country_fig, country_data = create_country_analysis(filtered_df)
                st.plotly_chart(country_fig, use_container_width=True)
            
            # Top countries table
            st.subheader("Top Performing Countries")
            st.dataframe(
                country_data.head(10).reset_index(drop=True).style.format({
                    'clicks': '{:,.0f}',
                    'impressions': '{:,.0f}',
                    'ctr': '{:.2%}',
                    'position': '{:.1f}'
                }),
                use_container_width=True
            )
        
        
        with st.expander("🌎 Spanish-Speaking Markets Analysis", expanded=False):
            st.markdown('<h2 class="section-header">Spanish Market Performance</h2>', unsafe_allow_html=True)
            with st.spinner("Analyzing Spanish-speaking markets..."):
                spanish_results = create_spanish_market_analysis(filtered_df)
            
            if spanish_results[0] is not None:
                spanish_metrics, row_metrics, spanish_top_queries, spanish_top_pages, spanish_by_country = spanish_results
                
                st.subheader("Spanish-Speaking Countries vs Rest of World")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="branded-section">', unsafe_allow_html=True)
                    st.write("### 🇪🇸 Spanish-Speaking Markets")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Total Clicks", f"{spanish_metrics['clicks']:,.0f}")
                        st.metric("Unique Queries", f"{spanish_metrics['unique_queries']:,}")
                    with metric_col2:
                        st.metric("Total Impressions", f"{spanish_metrics['impressions']:,.0f}")
                        st.metric("Unique Pages", f"{spanish_metrics['unique_pages']:,}")
                    with metric_col3:
                        st.metric("Avg CTR", f"{spanish_metrics['avg_ctr']:.2%}")
                        st.metric("Avg Position", f"{spanish_metrics['avg_position']:.1f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                total_clicks = spanish_metrics['clicks'] + row_metrics['clicks']
                spanish_share = (spanish_metrics['clicks'] / total_clicks * 100) if total_clicks > 0 else 0
                
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.write(f"**Market Share:** Spanish-speaking countries represent **{spanish_share:.1f}%** of total clicks")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.subheader("Performance by Spanish-Speaking Country")
                
                fig_spanish_countries = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Clicks by Country', 'Impressions by Country', 
                                'CTR by Country', 'Average Position by Country'),
                    specs=[[{"type": "bar"}, {"type": "bar"}],
                        [{"type": "bar"}, {"type": "bar"}]]
                )
                
                top_spanish_countries = spanish_by_country.head(10)
                
                fig_spanish_countries.add_trace(
                    go.Bar(x=top_spanish_countries['country'], y=top_spanish_countries['clicks'], 
                        name='Clicks', marker_color='#3498db'),
                    row=1, col=1
                )
                
                fig_spanish_countries.add_trace(
                    go.Bar(x=top_spanish_countries['country'], y=top_spanish_countries['impressions'], 
                        name='Impressions', marker_color='#2ecc71'),
                    row=1, col=2
                )
                
                fig_spanish_countries.add_trace(
                    go.Bar(x=top_spanish_countries['country'], y=top_spanish_countries['ctr'], 
                        name='CTR', marker_color='#e74c3c'),
                    row=2, col=1
                )
                
                fig_spanish_countries.add_trace(
                    go.Bar(x=top_spanish_countries['country'], y=top_spanish_countries['position'], 
                        name='Position', marker_color='#f39c12'),
                    row=2, col=2
                )
                
                fig_spanish_countries.update_layout(height=800, showlegend=False, 
                                                title_text="<b>Spanish-Speaking Countries Performance</b>")
                fig_spanish_countries.update_xaxes(tickangle=45)
                
                st.plotly_chart(fig_spanish_countries, use_container_width=True)
                
                st.dataframe(
                    spanish_by_country.reset_index(drop=True).style.format({
                        'clicks': '{:,.0f}',
                        'impressions': '{:,.0f}',
                        'ctr': '{:.2%}',
                        'position': '{:.1f}'
                    }),
                    use_container_width=True,
                    height=400
                )
                
                tab1, tab2 = st.tabs(["🔍 Top Queries", "📄 Top Pages"])
                
                with tab1:
                    st.subheader("Top 30 Queries in Spanish-Speaking Markets")
                    st.dataframe(
                        spanish_top_queries[['query', 'clicks', 'impressions', 'ctr', 'position']].reset_index(drop=True)
                        .style.format({
                            'clicks': '{:,.0f}',
                            'impressions': '{:,.0f}',
                            'ctr': '{:.2%}',
                            'position': '{:.1f}'
                        }),
                        use_container_width=True,
                        height=600
                    )
                
                with tab2:
                    st.subheader("Top 30 Pages in Spanish-Speaking Markets")
                    st.dataframe(
                        spanish_top_pages[['page', 'clicks', 'impressions', 'ctr', 'position', 'unique_queries']].reset_index(drop=True)
                        .style.format({
                            'clicks': '{:,.0f}',
                            'impressions': '{:,.0f}',
                            'ctr': '{:.2%}',
                            'position': '{:.1f}',
                            'unique_queries': '{:,.0f}'
                        }),
                        use_container_width=True,
                        height=600
                    )
                
            else:
                st.info("No data available for Spanish-speaking countries in the current filter.")
        # --------------------------------------------------
        # Country → Blog Category Performance (INLINE DRILL-DOWN)
        # --------------------------------------------------
        st.markdown(
            '<h2 class="section-header">Country → Blog Category Performance</h2>',
            unsafe_allow_html=True
        )

        if len(country_blog_df) > 0:

            selected_country_cb = st.selectbox(
                "Select Country for Blog Category Analysis",
                sorted(country_blog_df['country'].unique())
            )

            category_view = (
                country_blog_df[country_blog_df['country'] == selected_country_cb]
                .sort_values('clicks', ascending=False)
                .reset_index(drop=True)
            )

            st.subheader(f"Blog Categories Performance — {selected_country_cb.upper()}")

            # Loop each blog category and create expandable row
            for _, row in category_view.iterrows():

                with st.expander(
                    f"📂 {row['blog_category']}  |  "
                    f"Clicks: {row['clicks']:,}  |  "
                    f"Impr: {row['impressions']:,}  |  "
                    f"CTR: {row['ctr']:.2%}  |  "
                    f"Pos: {row['avg_position']:.1f}",
                    expanded=False
                ):
                    # Filter page-level data
                    pages_df = filtered_df[
                        (filtered_df['country'] == selected_country_cb) &
                        (filtered_df['blog_category'] == row['blog_category'])
                    ]

                    if len(pages_df) == 0:
                        st.info("No pages found for this blog category.")
                    else:
                        top_pages = (
                            pages_df
                            .groupby('page')
                            .agg(
                                clicks=('clicks', 'sum'),
                                impressions=('impressions', 'sum'),
                                avg_position=('position', 'mean')
                            )
                            .reset_index()
                        )

                        top_pages['ctr'] = (
                            top_pages['clicks'] / top_pages['impressions']
                        )

                        top_pages = (
                            top_pages
                            .sort_values('clicks', ascending=False)
                            .head(5)
                            .reset_index(drop=True)
                        )

                        st.markdown("**Top 5 Pages**")

                        st.dataframe(
                            top_pages.rename(columns={
                                'page': 'Page URL',
                                'avg_position': 'Avg Position'
                            }).style.format({
                                'clicks': '{:,.0f}',
                                'impressions': '{:,.0f}',
                                'ctr': '{:.2%}',
                                'Avg Position': '{:.1f}'
                            }),
                            use_container_width=True
                        )

        else:
            st.info("No blog category data available.")


        # Trending Analysis
        st.markdown('<h2 class="section-header">Trending Queries Analysis</h2>', unsafe_allow_html=True)
        with st.spinner("Analyzing trends..."):
            winners, losers = create_trending_analysis(filtered_df)
        
        if winners is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top Improving Queries")
                if len(winners) > 0:
                    winners_display = winners.reset_index()
                    winners_display['click_change_display'] = winners_display['click_pct_change'].apply(
                        lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%" if pd.notna(x) else "New"
                    )
                    winners_display['position_trend'] = winners_display['position_change'].apply(
                        lambda x: f"↑{abs(x):.1f}" if x < -0.5 else f"↓{x:.1f}" if x > 0.5 else "→" if pd.notna(x) else "New"
                    )
                    st.dataframe(
                        winners_display[['query', 'clicks_p1', 'clicks_p2', 'click_change_display', 'position_p1', 'position_p2', 'position_trend', 'improvement_score']].rename(columns={
                            'clicks_p1': 'Clicks (Old)',
                            'clicks_p2': 'Clicks (New)',
                            'click_change_display': 'Click Change',
                            'position_p1': 'Pos (Old)',
                            'position_p2': 'Pos (New)',
                            'position_trend': 'Pos Trend',
                            'improvement_score': 'Score'
                        }).reset_index(drop=True).style.format({
                            'Clicks (Old)': '{:.0f}',
                            'Clicks (New)': '{:.0f}',
                            'Pos (Old)': '{:.1f}',
                            'Pos (New)': '{:.1f}',
                            'Score': '{:.1f}'
                        }),
                        height=400,
                        use_container_width=True
                    )
                else:
                    st.info("No improving queries found")
            
            with col2:
                st.subheader("Declining Queries")
                if len(losers) > 0:
                    losers_display = losers.reset_index()
                    losers_display['click_change_display'] = losers_display['click_pct_change'].apply(
                        lambda x: f"{x:.1f}%" if pd.notna(x) else "Lost"
                    )
                    losers_display['position_trend'] = losers_display['position_change'].apply(
                        lambda x: f"↑{abs(x):.1f}" if x < -0.5 else f"↓{x:.1f}" if x > 0.5 else "→" if pd.notna(x) else "Lost"
                    )
                    st.dataframe(
                        losers_display[['query', 'clicks_p1', 'clicks_p2', 'click_change_display', 'position_p1', 'position_p2', 'position_trend', 'improvement_score']].rename(columns={
                            'clicks_p1': 'Clicks (Old)',
                            'clicks_p2': 'Clicks (New)',
                            'click_change_display': 'Click Change',
                            'position_p1': 'Pos (Old)',
                            'position_p2': 'Pos (New)',
                            'position_trend': 'Pos Trend',
                            'improvement_score': 'Score'
                        }).reset_index(drop=True).style.format({
                            'Clicks (Old)': '{:.0f}',
                            'Clicks (New)': '{:.0f}',
                            'Pos (Old)': '{:.1f}',
                            'Pos (New)': '{:.1f}',
                            'Score': '{:.1f}'
                        }),
                        height=400,
                        use_container_width=True
                    )
                else:
                    st.info("No declining queries found")
        else:
            st.info("Need at least 8 days of data for trend analysis.")
        
        # Content Gap Analysis - FIXED to only show when there's data
        with st.spinner("Identifying content gaps..."):
            content_gaps = extract_content_gaps(filtered_df)
        
        if len(content_gaps) > 0:
            st.markdown('<h2 class="section-header">Content Gap Opportunities</h2>', unsafe_allow_html=True)
            st.subheader("High Volume, Low Performance Queries")
            st.write("These queries have high impressions but poor performance - perfect for content optimization:")
            
            gap_display = content_gaps.copy()
            gap_display['opportunity_score'] = gap_display['impressions'] * (1 - gap_display['ctr'])
            
            st.dataframe(
                gap_display[['query', 'impressions', 'clicks', 'ctr', 'position', 'opportunity_score']].reset_index(drop=True).style.format({
                    'impressions': '{:,.0f}',
                    'clicks': '{:,.0f}',
                    'ctr': '{:.2%}',
                    'position': '{:.1f}',
                    'opportunity_score': '{:.0f}'
                }),
                use_container_width=True
            )
            
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            st.write("**Recommendations:**")
            st.write("• Create comprehensive content targeting these high-impression queries")
            st.write("• Optimize existing pages to improve CTR and position")
            st.write("• Consider creating topic clusters around these themes")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Page Performance Analysis - UPDATED with filtering
        st.markdown('<h2 class="section-header">Page Performance Analysis</h2>', unsafe_allow_html=True)
        with st.spinner("Analyzing page performance..."):
            page_perf_all, page_perf_filtered, page_perf_datasheets = create_page_performance_analysis(filtered_df)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📄 All Pages (Top 50)", "🚫 Filtered Pages (No Product Downloads)", "📊 Data Sheets Only"])
        
        with tab1:
            st.subheader("Top 50 Performing Pages (All)")
            st.dataframe(
                page_perf_all[['page', 'clicks', 'impressions', 'ctr', 'position', 'unique_queries']]
                .reset_index(drop=True).style.format({
                    'clicks': '{:,.0f}',
                    'impressions': '{:,.0f}',
                    'ctr': '{:.2%}',
                    'position': '{:.1f}',
                    'unique_queries': '{:,.0f}'
                }),
                use_container_width=True,
                height=600
            )
        
        with tab2:
            st.subheader("Top 50 Performing Pages (Excluding Product Downloads)")
            st.info("This view excludes pages containing 'product-downloads' in the URL for cleaner analysis.")
            st.dataframe(
                page_perf_filtered[['page', 'clicks', 'impressions', 'ctr', 'position', 'unique_queries']]
                .reset_index(drop=True).style.format({
                    'clicks': '{:,.0f}',
                    'impressions': '{:,.0f}',
                    'ctr': '{:.2%}',
                    'position': '{:.1f}',
                    'unique_queries': '{:,.0f}'
                }),
                use_container_width=True,
                height=600
            )
        
        with tab3:
            st.subheader("Data Sheets Performance (/resources/data-sheets/)")
            st.info("This view shows only pages containing '/resources/data-sheets/' in the URL.")
            if len(page_perf_datasheets) > 0:
                st.dataframe(
                    page_perf_datasheets[['page', 'clicks', 'impressions', 'ctr', 'position', 'unique_queries']]
                    .reset_index(drop=True).style.format({
                        'clicks': '{:,.0f}',
                        'impressions': '{:,.0f}',
                        'ctr': '{:.2%}',
                        'position': '{:.1f}',
                        'unique_queries': '{:,.0f}'
                    }),
                    use_container_width=True,
                    height=600
                )
                # Show summary stats
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.write("**Data Sheets Summary:**")
                st.write(f"• Total Data Sheet Pages: **{len(page_perf_datasheets)}**")
                st.write(f"• Total Clicks: **{page_perf_datasheets['clicks'].sum():,.0f}**")
                st.write(f"• Total Impressions: **{page_perf_datasheets['impressions'].sum():,.0f}**")
                st.write(f"• Average CTR: **{page_perf_datasheets['ctr'].mean():.2%}**")
                st.write(f"• Average Position: **{page_perf_datasheets['position'].mean():.1f}**")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No data sheets pages found in the current filter.")
        
        # Time Series Analysis - ENHANCED WITH WEEKLY VIEW AND NEW METRICS
        st.markdown('<h2 class="section-header">Performance Over Time</h2>', unsafe_allow_html=True)
        
        # Create tabs for daily and weekly views
        time_tab1, time_tab2 = st.tabs(["📅 Daily View", "📊 Weekly View"])
        
        with time_tab1:
            # Daily aggregation with new metrics
            daily_perf = filtered_df.groupby('date').agg({
                'clicks': 'sum',
                'impressions': 'sum',
                'position': 'mean',
                'query': 'nunique',
                'page': 'nunique'
            }).reset_index()
            daily_perf['ctr'] = daily_perf['clicks'] / daily_perf['impressions']
            daily_perf = daily_perf.rename(columns={
                'query': 'unique_queries',
                'page': 'unique_urls'
            })
            
            fig_daily = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Daily Clicks', 'Daily Impressions', 
                            'Daily CTR', 'Daily Avg Position',
                            'Unique Ranking Queries', 'Unique Ranking URLs'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig_daily.add_trace(
                go.Scatter(x=daily_perf['date'], y=daily_perf['clicks'], 
                        mode='lines+markers', name='Clicks', line=dict(color='#3498db', width=2)),
                row=1, col=1
            )
            
            fig_daily.add_trace(
                go.Scatter(x=daily_perf['date'], y=daily_perf['impressions'], 
                        mode='lines+markers', name='Impressions', line=dict(color='#2ecc71', width=2)),
                row=1, col=2
            )
            
            fig_daily.add_trace(
                go.Scatter(x=daily_perf['date'], y=daily_perf['ctr'], 
                        mode='lines+markers', name='CTR', line=dict(color='#e74c3c', width=2)),
                row=2, col=1
            )
            
            fig_daily.add_trace(
                go.Scatter(x=daily_perf['date'], y=daily_perf['position'], 
                          mode='lines+markers', name='Position', line=dict(color='#f39c12', width=2)),
                row=2, col=2
            )
            
            fig_daily.add_trace(
                go.Scatter(x=daily_perf['date'], y=daily_perf['unique_queries'], 
                          mode='lines+markers', name='Unique Queries', line=dict(color='#9b59b6', width=2)),
                row=3, col=1
            )
            
            fig_daily.add_trace(
                go.Scatter(x=daily_perf['date'], y=daily_perf['unique_urls'], 
                          mode='lines+markers', name='Unique URLs', line=dict(color='#e67e22', width=2)),
                row=3, col=2
            )
            
            fig_daily.update_layout(height=900, showlegend=False, title_text="<b>Daily Performance Trends</b>")
            st.plotly_chart(fig_daily, use_container_width=True)
            
            # Show correlation insights
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.write("**Daily Insights:**")
            st.write(f"• Average unique queries per day: **{daily_perf['unique_queries'].mean():,.0f}**")
            st.write(f"• Average unique URLs per day: **{daily_perf['unique_urls'].mean():,.0f}**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with time_tab2:
            # Weekly aggregation
            weekly_df = filtered_df.copy()
            weekly_df['week'] = weekly_df['date'].dt.to_period('W').apply(lambda r: r.start_time)
            
            weekly_perf = weekly_df.groupby('week').agg({
                'clicks': 'sum',
                'impressions': 'sum',
                'position': 'mean',
                'query': 'nunique',
                'page': 'nunique'
            }).reset_index()
            weekly_perf['ctr'] = weekly_perf['clicks'] / weekly_perf['impressions']
            weekly_perf = weekly_perf.rename(columns={
                'query': 'unique_queries',
                'page': 'unique_urls'
            })
            
            fig_weekly = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Weekly Clicks', 'Weekly Impressions', 
                              'Weekly CTR', 'Weekly Avg Position',
                              'Unique Ranking Queries', 'Unique Ranking URLs'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                      [{"secondary_y": False}, {"secondary_y": False}],
                      [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig_weekly.add_trace(
                go.Scatter(x=weekly_perf['week'], y=weekly_perf['clicks'], 
                        mode='lines+markers', name='Clicks', line=dict(color='#3498db', width=2),
                        marker=dict(size=8)),
                row=1, col=1
            )
            
            fig_weekly.add_trace(
                go.Scatter(x=weekly_perf['week'], y=weekly_perf['impressions'], 
                        mode='lines+markers', name='Impressions', line=dict(color='#2ecc71', width=2),
                        marker=dict(size=8)),
                row=1, col=2
            )
            
            fig_weekly.add_trace(
                go.Scatter(x=weekly_perf['week'], y=weekly_perf['ctr'], 
                        mode='lines+markers', name='CTR', line=dict(color='#e74c3c', width=2),
                        marker=dict(size=8)),
                row=2, col=1
            )
            
            fig_weekly.add_trace(
                go.Scatter(x=weekly_perf['week'], y=weekly_perf['position'], 
                          mode='lines+markers', name='Position', line=dict(color='#f39c12', width=2),
                          marker=dict(size=8)),
                row=2, col=2
            )
            
            fig_weekly.add_trace(
                go.Scatter(x=weekly_perf['week'], y=weekly_perf['unique_queries'], 
                        mode='lines+markers', name='Unique Queries', line=dict(color='#9b59b6', width=2),
                        marker=dict(size=8)),
                row=3, col=1
            )
            
            fig_weekly.add_trace(
                go.Scatter(x=weekly_perf['week'], y=weekly_perf['unique_urls'], 
                        mode='lines+markers', name='Unique URLs', line=dict(color='#e67e22', width=2),
                        marker=dict(size=8)),
                row=3, col=2
            )
            
            fig_weekly.update_layout(height=900, showlegend=False, title_text="<b>Weekly Performance Trends</b>")
            st.plotly_chart(fig_weekly, use_container_width=True)
            
            # Weekly Click Drop Analysis
            st.markdown('<h3 class="section-header">Weekly Click Changes & Query Analysis</h3>', unsafe_allow_html=True)

            if len(weekly_perf) > 1:
                # Calculate week-over-week changes
                for i in range(1, len(weekly_perf)):
                    prev_week = weekly_perf.iloc[i-1]
                    curr_week = weekly_perf.iloc[i]
                    
                    click_change = float(curr_week['clicks'] - prev_week['clicks'])
                    click_pct_change = (click_change / prev_week['clicks'] * 100) if prev_week['clicks'] > 0 else 0
                    
                    # Display every week's comparison (both drops and gains)
                    with st.expander(
                        f"📊 Week {curr_week['week'].strftime('%b %d')} vs Week {prev_week['week'].strftime('%b %d')} "
                        f"({'📉 Drop' if click_change < 0 else '📈 Gain'}: {click_change:+,.0f} clicks, {click_pct_change:+.1f}%)",
                        expanded=(click_change < 0)  # Auto-expand only drops
                    ):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Previous Week", f"{prev_week['clicks']:,.0f}")
                        with col2:
                            st.metric("Current Week", f"{curr_week['clicks']:,.0f}", 
                                    delta=f"{click_change:,.0f}")
                        with col3:
                            st.metric("Change %", f"{click_pct_change:+.1f}%")
                        
                        # Calculate week boundaries
                        curr_week_start = curr_week['week']
                        curr_week_end = curr_week_start + timedelta(days=6)
                        prev_week_start = prev_week['week']
                        prev_week_end = prev_week_start + timedelta(days=6)
                        
                        # Get query-level data for both weeks
                        prev_week_queries = filtered_df[
                            (filtered_df['date'] >= prev_week_start) & 
                            (filtered_df['date'] <= prev_week_end)
                        ].groupby('query').agg({
                            'clicks': 'sum', 
                            'impressions': 'sum', 
                            'position': 'mean'
                        }).reset_index()
                        
                        curr_week_queries = filtered_df[
                            (filtered_df['date'] >= curr_week_start) & 
                            (filtered_df['date'] <= curr_week_end)
                        ].groupby('query').agg({
                            'clicks': 'sum', 
                            'impressions': 'sum', 
                            'position': 'mean'
                        }).reset_index()
                        
                        # Merge and calculate changes
                        query_comparison = pd.merge(
                            prev_week_queries, 
                            curr_week_queries, 
                            on='query', 
                            how='outer', 
                            suffixes=('_prev', '_curr')
                        ).fillna(0)
                        
                        query_comparison['click_change'] = query_comparison['clicks_curr'] - query_comparison['clicks_prev']
                        query_comparison['click_pct_change'] = np.where(
                            query_comparison['clicks_prev'] > 0,
                            (query_comparison['click_change'] / query_comparison['clicks_prev']) * 100,
                            np.nan
                        )
                        query_comparison['position_change'] = query_comparison['position_curr'] - query_comparison['position_prev']
                        
                        # Always show BOTH top losers and top gainers for every week
                        top_losers = query_comparison[
                            (query_comparison['click_change'] < 0) & 
                            (query_comparison['clicks_prev'] >= 3)
                        ].sort_values('click_change', ascending=True).head(20)

                        top_gainers = query_comparison[
                            (query_comparison['click_change'] > 0) & 
                            (query_comparison['clicks_curr'] >= 3)
                        ].sort_values('click_change', ascending=False).head(20)

                        # Show losers first
                        if len(top_losers) > 0:
                            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                            st.write(f"**Top Queries with Decreased Clicks** ({len(top_losers)} queries shown)")
                            
                            display_losers = top_losers[['query', 'clicks_prev', 'clicks_curr', 'click_change', 
                                                        'click_pct_change', 'position_prev', 'position_curr', 
                                                        'position_change']].copy()
                            
                            st.dataframe(
                                display_losers.rename(columns={
                                    'query': 'Query',
                                    'clicks_prev': 'Prev Clicks',
                                    'clicks_curr': 'Curr Clicks',
                                    'click_change': 'Change',
                                    'click_pct_change': 'Change %',
                                    'position_prev': 'Prev Pos',
                                    'position_curr': 'Curr Pos',
                                    'position_change': 'Pos Δ'
                                }).reset_index(drop=True).style.format({
                                    'Prev Clicks': '{:.0f}',
                                    'Curr Clicks': '{:.0f}',
                                    'Change': '{:+.0f}',
                                    'Change %': '{:+.1f}%',
                                    'Prev Pos': '{:.1f}',
                                    'Curr Pos': '{:.1f}',
                                    'Pos Δ': '{:+.1f}'
                                }),
                                use_container_width=True,
                                height=400
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.info("No queries with decreased clicks found.")

                        # Show gainers second
                        if len(top_gainers) > 0:
                            st.markdown('<div class="branded-section">', unsafe_allow_html=True)
                            st.write(f"**Top Queries with Increased Clicks** ({len(top_gainers)} queries shown)")
                            
                            display_gainers = top_gainers[['query', 'clicks_prev', 'clicks_curr', 'click_change', 
                                                        'click_pct_change', 'position_prev', 'position_curr', 
                                                        'position_change']].copy()
                            
                            st.dataframe(
                                display_gainers.rename(columns={
                                    'query': 'Query',
                                    'clicks_prev': 'Prev Clicks',
                                    'clicks_curr': 'Curr Clicks',
                                    'click_change': 'Change',
                                    'click_pct_change': 'Change %',
                                    'position_prev': 'Prev Pos',
                                    'position_curr': 'Curr Pos',
                                    'position_change': 'Pos Δ'
                                }).reset_index(drop=True).style.format({
                                    'Prev Clicks': '{:.0f}',
                                    'Curr Clicks': '{:.0f}',
                                    'Change': '{:+.0f}',
                                    'Change %': '{:+.1f}%',
                                    'Prev Pos': '{:.1f}',
                                    'Curr Pos': '{:.1f}',
                                    'Pos Δ': '{:+.1f}'
                                }),
                                use_container_width=True,
                                height=400
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.info("No queries with increased clicks found.")
            
            # Show weekly insights
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.write("**Weekly Insights:**")
            st.write(f"• Average unique queries per week: **{weekly_perf['unique_queries'].mean():,.0f}**")
            st.write(f"• Average unique URLs per week: **{weekly_perf['unique_urls'].mean():,.0f}**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Export functionality
        st.markdown('<h2 class="section-header">Export Analysis</h2>', unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("Export Opportunity Data"):
                if 'opp_data' in locals() and len(opp_data) > 0:
                    csv = opp_data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="opportunity_analysis.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No opportunity data to export")
        
        with col2:
            if st.button("Export Branded Analysis"):
                if 'branded_opp' in locals() and len(branded_opp) > 0:
                    # Combine branded opportunities and top performers
                    branded_combined = pd.concat([
                        branded_opp.assign(type='Opportunity'),
                        branded_top.assign(type='Top Performer')
                    ], ignore_index=True)
                    csv = branded_combined.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="branded_analysis.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No branded data to export")
        
        with col3:
            if st.button("Export Non-Branded Analysis"):
                if 'non_branded_opp' in locals() and len(non_branded_opp) > 0:
                    # Combine non-branded opportunities and top performers
                    non_branded_combined = pd.concat([
                        non_branded_opp.assign(type='Opportunity'),
                        non_branded_top.assign(type='Top Performer')
                    ], ignore_index=True)
                    csv = non_branded_combined.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="non_branded_analysis.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No non-branded data to export")
        
        with col4:
            if st.button("Export Page Performance"):
                csv = page_perf_filtered.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="page_performance_filtered.csv",
                    mime="text/csv"
                )
        
        with col5:
            if st.button("Export Spanish Markets"):
                if 'spanish_results' in locals() and spanish_results[0] is not None:
                    # Create comprehensive Spanish markets export
                    spanish_export = pd.concat([
                        spanish_top_queries.assign(type='Top Query'),
                        spanish_top_pages.assign(type='Top Page'),
                        spanish_by_country.assign(type='Country')
                    ], ignore_index=True)
                    csv = spanish_export.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="spanish_markets_analysis.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No Spanish markets data to export")
    except KeyError as e:
        st.error("Missing environment variable for Supabase.")
        st.code(str(e))
        st.info("Ensure SUPABASE_URL and SUPABASE_ANON_KEY are set.")
    
    except Exception as e:
        st.error(f"An error occurred while loading or processing the data: {e}")
    """ 
    except FileNotFoundError:
        st.error(f"Error: The data file was not found at the path: `{data_path}`")
        st.info("Please make sure the data file exists in the specified directory and has the correct name.")
        st.warning("The application expects the data file to be in a subfolder named 'Data' relative to the script.")
    """

if __name__ == "__main__":
    main()