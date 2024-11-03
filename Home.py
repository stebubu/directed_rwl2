# app/Home.py
import streamlit as st
import folium
from streamlit_folium import folium_static
import rasterio
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Geospatial Analysis Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Function to create date slider
def create_date_slider():
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    
    selected_date = st.select_slider(
        "Select a date",
        options=dates,
        format_func=lambda x: x.strftime("%Y-%m-%d")
    )
    return selected_date

# Function to create map viewer
def create_map_viewer():
    uploaded_file = st.file_uploader("Upload a GeoTIFF file", type=['tif', 'tiff'])
    
    if uploaded_file is not None:
        with rasterio.open(uploaded_file) as src:
            bounds = src.bounds
            data = src.read(1)
            
            center_lat = (bounds.bottom + bounds.top) / 2
            center_lon = (bounds.left + bounds.right) / 2
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=10
            )
            
            folium.raster_layers.ImageOverlay(
                data,
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                colormap=lambda x: (1, 0, 0, x),
                name='Raster'
            ).add_to(m)
            
            folium.LayerControl().add_to(m)
            
            folium_static(m)
    else:
        st.info("Please upload a GeoTIFF file to view the map")

# Function to create time series plot
def create_time_series():
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start date",
            datetime(2020, 1, 1)
        )
    with col2:
        end_date = st.date_input(
            "End date",
            datetime(2024, 12, 31)
        )

    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    values = np.random.normal(0, 1, size=len(dates))
    df = pd.DataFrame({
        'Date': dates,
        'Value': values
    })

    fig = px.line(
        df,
        x='Date',
        y='Value',
        title='Time Series Data'
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

# Main navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Overview", "Analysis 1", "Analysis 2", "Analysis 3"])

if page == "Overview":
    st.title("Geospatial Analysis Dashboard")
    st.write("Select different pages from the sidebar to view different analyses.")
    
    # Create three columns for the components
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        st.subheader("Date Selection")
        selected_date = create_date_slider()
        
    with col2:
        st.subheader("Map View")
        create_map_viewer()
        
    with col3:
        st.subheader("Time Series")
        create_time_series()

elif page == "Analysis 1":
    st.title("Analysis 1")
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["Date Selection", "Map View", "Time Series"])
    
    with tab1:
        selected_date = create_date_slider()
    
    with tab2:
        create_map_viewer()
    
    with tab3:
        create_time_series()

elif page == "Analysis 2":
    st.title("Analysis 2")
    
    # Vertical layout
    st.subheader("1. Select Date")
    selected_date = create_date_slider()
    
    st.subheader("2. Geospatial View")
    create_map_viewer()
    
    st.subheader("3. Temporal Analysis")
    create_time_series()

elif page == "Analysis 3":
    st.title("Analysis 3")
    
    # Two-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Date Selection")
        selected_date = create_date_slider()
        
        st.subheader("Time Series")
        create_time_series()
    
    with col2:
        st.subheader("Map View")
        create_map_viewer()
