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

# Function to create map viewer with Radar and Flood layers
def create_map_viewer():
    st.subheader("Radar Rainfall Intensity and Flood Area Overlay")
    
    # Set map to Google Satellite centered on Rimini
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google"
    )
    
    # Overlay Radar Rainfall Intensity
    radar_data = generate_random_geotiff_data(shape=(100, 100))
    folium.raster_layers.ImageOverlay(
        radar_data,
        bounds=bounds,
        colormap=lambda x: (0, 0, 1, x),  # blue color for rainfall intensity
        name="Radar Rainfall Intensity"
    ).add_to(m)
    
    # Optional: Add Flood Overlay if button is clicked
    if st.button("Generate Flood Area Overlay"):
        flood_data = generate_random_geotiff_data(shape=(100, 100))  # Random flood data
        folium.raster_layers.ImageOverlay(
            flood_data,
            bounds=bounds,
            colormap=lambda x: (1, 0, 0, x),  # red color for flood (water depth)
            name="Flood Area"
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
    dates = pd.date_range(start=selected_date - timedelta(days=30), end=selected_date, freq='D')
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
page = st.sidebar.radio("Select a page:", ["Realtime Pluvial", "Analysis 1", "Analysis 2", "Analysis 3"])

if page == "Realtime Pluvial":
    st.title("Realtime Pluvial Dashboard")
    st.write("Select different pages from the sidebar to view different analyses.")
    
    # Create three columns for the components
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        st.subheader("Date Selection")
        selected_date = create_date_slider()
        
        # Dropdown to select point for time series
        point_selection = st.selectbox(
            "Select a point to view time series:",
            options=[f"Point {i+1}" for i in range(len(random_points))],
            index=0
        )
        selected_point_id = int(point_selection.split()[1])  # Extract point ID from dropdown text
        
    with col2:
        st.subheader("Map View")
        create_map_viewer()
        
    with col3:
        st.subheader("Time Series")
        create_time_series(selected_date, selected_point_id)


