# app/Home.py
import streamlit as st
import folium
from streamlit_folium import folium_static
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

# Center of Rimini, Italy
center_lat, center_lon = 44.0633, 12.5808
bounds = [[center_lat - 0.09, center_lon - 0.09], [center_lat + 0.09, center_lon + 0.09]]

# Generate static random points in Rimini area
random_points = [(center_lat + random.uniform(-0.09, 0.09), center_lon + random.uniform(-0.09, 0.09)) for _ in range(5)]

# Function to create a date slider with specific ranges
def create_date_slider(start, end):
    dates = pd.date_range(start=start, end=end, freq='H')
    selected_date = st.select_slider(
        "Select a forecast time",
        options=dates,
        format_func=lambda x: x.strftime("%Y-%m-%d %H:%M")
    )
    return selected_date

# Function to generate random GeoTIFF-like data based on scenario
def generate_random_geotiff_data(shape=(100, 100), intensity=1.0):
    return np.random.random(shape) * intensity

# Function to create map viewer with dynamic component names
def create_map_viewer(radar_name, radar_intensity, flood_name, flood_intensity):
    st.subheader(f"{radar_name} and {flood_name} Overlay")
    
    # Set map to Google Satellite centered on Rimini
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google"
    )
    
    # Overlay radar with specified name and intensity
    radar_data = generate_random_geotiff_data(shape=(100, 100), intensity=radar_intensity)
    folium.raster_layers.ImageOverlay(
        radar_data,
        bounds=bounds,
        colormap=lambda x: (0, 0, 1, x),
        name=radar_name
    ).add_to(m)
    
    # Optional: Add flood overlay with specified name and intensity if button is clicked
    if st.button(f"Generate {flood_name} Overlay"):
        flood_data = generate_random_geotiff_data(shape=(100, 100), intensity=flood_intensity)
        folium.raster_layers.ImageOverlay(
            flood_data,
            bounds=bounds,
            colormap=lambda x: (1, 0, 0, x),
            name=flood_name
        ).add_to(m)
    
    # Add static random points to the map with labels
    for i, (lat, lon) in enumerate(random_points):
        folium.Marker(
            location=[lat, lon],
            popup=f"Point {i+1}",
            tooltip="Click to select"
        ).add_to(m)

    folium.LayerControl().add_to(m)
    folium_static(m)

# Function to create time series plot for a specific point
def create_time_series(selected_date, point_id):
    dates = pd.date_range(start=selected_date - timedelta(days=2), end=selected_date, freq='H')
    values = np.random.normal(0, 1, size=len(dates)) + point_id  # Vary values slightly by point
    df = pd.DataFrame({
        'Date': dates,
        'Value': values
    })

    fig = px.line(
        df,
        x='Date',
        y='Value',
        title=f'Time Series Data for Point {point_id}'
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

# Main navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Realtime Pluvial", "Forecast Pluvial", "Coastal Flooding Forecast", "Coastal NowCasting"])

if page == "Realtime Pluvial":
    st.title("Realtime Pluvial Dashboard")
    
    scenario = st.selectbox(
        "Select a cumulative rainfall scenario:",
        options=["1h", "3h", "6h", "12h"],
        index=0
    )
    
    # Set intensities based on the selected scenario
    radar_intensity = 1.0 + 0.5 * ["1h", "3h", "6h", "12h"].index(scenario)
    flood_intensity = 0.5 + 0.3 * ["1h", "3h", "6h", "12h"].index(scenario)

    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        selected_date = create_date_slider(datetime(2020, 1, 1), datetime(2024, 12, 31))
        point_selection = st.selectbox("Select a point to view time series:", [f"Point {i+1}" for i in range(len(random_points))], index=0)
        selected_point_id = int(point_selection.split()[1])
        
    with col2:
        create_map_viewer("Radar Rainfall Intensity", radar_intensity, "Flood Area", flood_intensity)
        
    with col3:
        create_time_series(selected_date, selected_point_id)

elif page == "Forecast Pluvial":
    st.title("Forecast Pluvial")
    
    # Add cumulative rainfall scenario selection here
    scenario = st.selectbox(
        "Select a cumulative rainfall scenario:",
        options=["1h", "3h", "6h", "12h"],
        index=0
    )
    
    # Set intensities based on the selected scenario
    radar_intensity = 1.0 + 0.5 * ["1h", "3h", "6h", "12h"].index(scenario)
    flood_intensity = 0.5 + 0.3 * ["1h", "3h", "6h", "12h"].index(scenario)

    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        selected_date = create_date_slider(datetime.now(), datetime.now() + timedelta(hours=48))
        point_selection = st.selectbox("Select a point to view time series:", [f"Point {i+1}" for i in range(len(random_points))], index=0)
        selected_point_id = int(point_selection.split()[1])
        
    with col2:
        create_map_viewer("COSMO Forecast Rainfall", radar_intensity, "Flood Simulation", flood_intensity)
        
    with col3:
        create_time_series(selected_date, selected_point_id)

elif page == "Coastal Flooding Forecast":
    st.title("Coastal Flooding Forecast")
    
    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        selected_date = create_date_slider(datetime.now(), datetime.now() + timedelta(hours=48))
        point_selection = st.selectbox("Select a point to view time series:", [f"Point {i+1}" for i in range(len(random_points))], index=0)
        selected_point_id = int(point_selection.split()[1])
        
    with col2:
        create_map_viewer("SEA Model Forecast", radar_intensity=2.0, flood_name="Coastal Flood Simulation", flood_intensity=1.2)
        
    with col3:
        create_time_series(selected_date, selected_point_id)

elif page == "Coastal NowCasting":
    st.title("Coastal NowCasting")
    st.write("This page is under development.")

