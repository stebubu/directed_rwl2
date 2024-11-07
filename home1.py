# app/Home.py
import streamlit as st
import folium
from streamlit_folium import folium_static
import rasterio
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import random

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

# Function to generate random GeoTIFF-like data
def generate_random_geotiff_data(shape=(100, 100)):
    return np.random.random(shape)

# Function to create map viewer with random GeoTIFF data and Google Satellite background
def create_map_viewer():
    st.subheader("Radar Rainfall Intensity Overlay")
    
    # Generate random rainfall intensity data
    data = generate_random_geotiff_data()
    bounds = [[-90, -180], [90, 180]]  # Example global bounds for demo purposes
    
    # Set map to Google Satellite
    m = folium.Map(
        location=[0, 0],
        zoom_start=2,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google"
    )
    
    # Overlay random rainfall intensity raster
    folium.raster_layers.ImageOverlay(
        data,
        bounds=bounds,
        colormap=lambda x: (0, 0, 1, x),  # blue color for rainfall intensity
        name="Rainfall Intensity"
    ).add_to(m)

    # Adding random points to the map
    for _ in range(5):  # Adding five random points for demonstration
        lat, lon = random.uniform(-90, 90), random.uniform(-180, 180)
        folium.Marker([lat, lon], popup=f"Point at {lat:.2f}, {lon:.2f}").add_to(m)
    
    # Button to add flood overlay
    if st.button("Generate Flood Area Overlay"):
        flood_data = generate_random_geotiff_data()  # Random flood data
        folium.raster_layers.ImageOverlay(
            flood_data,
            bounds=bounds,
            colormap=lambda x: (0, 0, 1, x),  # different colormap for flood (water depth)
            name="Flood Area"
        ).add_to(m)

    folium.LayerControl().add_to(m)
    folium_static(m)

# Function to create time series plot for a point
def create_time_series(selected_date):
    dates = pd.date_range(start=selected_date - timedelta(days=30), end=selected_date, freq='D')
    values = np.random.normal(0, 1, size=len(dates))
    df = pd.DataFrame({
        'Date': dates,
        'Value': values
    })

    fig = px.line(
        df,
        x='Date',
        y='Value',
        title='Time Series Data for Selected Point'
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

# Main navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Realtime Pluvial", "Analysis 1", "Analysis 2", "Analysis 3"])

if page == "Realtime Pluvial":
    st.title("Realtime Pluvial Dashboard")
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
        create_time_series(selected_date)

elif page == "Analysis 1":
    st.title("Analysis 1")
    
    tab1, tab2, tab3 = st.tabs(["Date Selection", "Map View", "Time Series"])
    
    with tab1:
        selected_date = create_date_slider()
    
    with tab2:
        create_map_viewer()
    
    with tab3:
        create_time_series(selected_date)

elif page == "Analysis 2":
    st.title("Analysis 2")
    
    st.subheader("1. Select Date")
    selected_date = create_date_slider()
    
    st.subheader("2. Geospatial View")
    create_map_viewer()
    
    st.subheader("3. Temporal Analysis")
    create_time_series(selected_date)

elif page == "Analysis 3":
    st.title("Analysis 3")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Date Selection")
        selected_date = create_date_slider()
        
        st.subheader("Time Series")
        create_time_series(selected_date)
    
    with col2:
        st.subheader("Map View")
        create_map_viewer()
