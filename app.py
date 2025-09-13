import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import altair as alt
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file):
    """Load and preprocess GSC data"""
    df = pd.read_csv(uploaded_file)
    try:
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    except ValueError:
        try:
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        except ValueError:
            df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    
    df['ctr'] = df['ctr'].astype(float)
    df['position'] = df['position'].astype(float)
    return df

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
    """Analyze page-level performance"""
    page_perf = df.groupby('page').agg({
        'clicks': 'sum',
        'impressions': 'sum',
        'position': 'mean',
        'query': 'nunique'
    }).reset_index()
    
    page_perf['ctr'] = page_perf['clicks'] / page_perf['impressions']
    page_perf = page_perf.rename(columns={'query': 'unique_queries'})
    page_perf = page_perf.sort_values('clicks', ascending=False)
    
    return page_perf.head(20)

def main():
    st.markdown('<h1 class="main-header">GSC Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for file upload and filters
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Choose GSC CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load data
        with st.spinner("Loading data..."):
            df = load_data(uploaded_file)
        
        st.sidebar.success(f"Loaded {len(df):,} rows of data")
        
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
                
                # FIXED: Insights with proper filtering
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.subheader("Key Insights")
                
                # Calculate medians from the filtered data
                median_impressions = opp_data['impressions'].median()
                median_ctr = opp_data['ctr_calculated'].median()
                
                high_opp = opp_data[
                    (opp_data['impressions'] > median_impressions) &
                    (opp_data['ctr_calculated'] < median_ctr)
                ].sort_values('impressions', ascending=False)
                
                top_performers = opp_data[
                    (opp_data['impressions'] > median_impressions) &
                    (opp_data['ctr_calculated'] > median_ctr)
                ].sort_values('clicks', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Biggest Opportunities:** {len(high_opp)} queries with high impressions but low CTR")
                    if len(high_opp) > 0:
                        st.dataframe(
                            high_opp[['query', 'impressions', 'clicks', 'ctr_calculated', 'position']].style.format({
                                'impressions': '{:,.0f}',
                                'clicks': '{:,.0f}',
                                'ctr_calculated': '{:.2%}',
                                'position': '{:.1f}'
                            }),
                            height=400,
                            use_container_width=True
                        )
                    else:
                        st.info("No high-impression, low-CTR queries found in current filter.")
                
                with col2:
                    st.write(f"**Top Performers:** {len(top_performers)} high-performing queries")
                    if len(top_performers) > 0:
                        st.dataframe(
                            top_performers[['query', 'impressions', 'clicks', 'ctr_calculated', 'position']].style.format({
                                'impressions': '{:,.0f}',
                                'clicks': '{:,.0f}',
                                'ctr_calculated': '{:.2%}',
                                'position': '{:.1f}'
                            }),
                            height=400,
                            use_container_width=True
                        )
                    else:
                        st.info("No high-performing queries found in current filter.")
                
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
                country_data.head(10).style.format({
                    'clicks': '{:,.0f}',
                    'impressions': '{:,.0f}',
                    'ctr': '{:.2%}',
                    'position': '{:.1f}'
                }),
                use_container_width=True
            )
        
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
                        }).style.format({
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
                        }).style.format({
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
                gap_display[['query', 'impressions', 'clicks', 'ctr', 'position', 'opportunity_score']].style.format({
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
        
        # Page Performance Analysis
        st.markdown('<h2 class="section-header">Page Performance Analysis</h2>', unsafe_allow_html=True)
        with st.spinner("Analyzing page performance..."):
            page_perf = create_page_performance_analysis(filtered_df)
        
        st.subheader("Top Performing Pages")
        st.dataframe(
            page_perf[['page', 'clicks', 'impressions', 'ctr', 'position', 'unique_queries']].style.format({
                'clicks': '{:,.0f}',
                'impressions': '{:,.0f}',
                'ctr': '{:.2%}',
                'position': '{:.1f}',
                'unique_queries': '{:,.0f}'
            }),
            use_container_width=True
        )
        
        # Time Series Analysis
        st.markdown('<h2 class="section-header">Performance Over Time</h2>', unsafe_allow_html=True)
        daily_perf = filtered_df.groupby('date').agg({
            'clicks': 'sum',
            'impressions': 'sum',
            'position': 'mean'
        }).reset_index()
        daily_perf['ctr'] = daily_perf['clicks'] / daily_perf['impressions']
        
        fig_time = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Clicks', 'Daily Impressions', 'Daily CTR', 'Daily Avg Position'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig_time.add_trace(
            go.Scatter(x=daily_perf['date'], y=daily_perf['clicks'], 
                      mode='lines+markers', name='Clicks', line=dict(color='#3498db', width=2)),
            row=1, col=1
        )
        
        fig_time.add_trace(
            go.Scatter(x=daily_perf['date'], y=daily_perf['impressions'], 
                      mode='lines+markers', name='Impressions', line=dict(color='#2ecc71', width=2)),
            row=1, col=2
        )
        
        fig_time.add_trace(
            go.Scatter(x=daily_perf['date'], y=daily_perf['ctr'], 
                      mode='lines+markers', name='CTR', line=dict(color='#e74c3c', width=2)),
            row=2, col=1
        )
        
        fig_time.add_trace(
            go.Scatter(x=daily_perf['date'], y=daily_perf['position'], 
                      mode='lines+markers', name='Position', line=dict(color='#f39c12', width=2)),
            row=2, col=2
        )
        
        fig_time.update_layout(height=600, showlegend=False, title_text="<b>Performance Trends Over Time</b>")
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Export functionality
        st.markdown('<h2 class="section-header">Export Analysis</h2>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
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
                    st.warning("No content gaps to export")
        
        with col3:
            if st.button("Export Page Performance"):
                csv = page_perf.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="page_performance.csv",
                    mime="text/csv"
                )
    
    else:
        # Landing page with instructions
        st.markdown("""
        ## Welcome to the GSC Analytics Dashboard
        
        This dashboard helps you analyze your Google Search Console data to:
        
        - **Identify keyword opportunities** using advanced quadrant analysis
        - **Track trending queries** to spot winners and losers
        - **Analyze geographic performance** across different countries
        - **Find content gaps** with high potential
        - **Evaluate page performance** to optimize your best content
        - **Monitor trends** over time
        
        ### To get started:
        1. Upload your GSC CSV file using the sidebar
        2. Apply filters to focus your analysis
        3. Explore the insights and download recommendations
        
        ### Expected CSV Format:
        ```
        date, country, query, page, clicks, impressions, ctr, position
        ```
        
        Your data should include the last 10 days with country-level breakdowns for comprehensive analysis.
        """)

if __name__ == "__main__":
    main()
