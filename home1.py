# app/Home.py
import streamlit as st
import folium
from streamlit_folium import folium_static
from folium.plugins import Draw
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import random
import requests
import tempfile
import os
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import xarray as xr
import pytz
from branca.colormap import linear
from matplotlib import cm, colors

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

# Function to create map viewer with editable polyline
def create_map_viewer_with_barrier(radar_name, radar_intensity, flood_name, flood_intensity):
    st.subheader(f"{radar_name} and {flood_name} Overlay with Editable Barrier")

    # Set up map centered on Rimini with Google Satellite view
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google"
    )

    # Add radar overlay with specified intensity
    radar_data = generate_random_geotiff_data(shape=(100, 100), intensity=radar_intensity)
    folium.raster_layers.ImageOverlay(
        radar_data,
        bounds=bounds,
        colormap=lambda x: (0, 0, 1, x),
        name=radar_name
    ).add_to(m)

    # Optional: Add flood overlay with specified intensity if button is clicked
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

    # Add an editable polyline barrier with the Draw plugin
    draw = Draw(
        draw_options={
            'polyline': {'shapeOptions': {'color': 'red'}},
            'polygon': False,
            'rectangle': False,
            'circle': False,
            'marker': False,
            'circlemarker': False
        },
        edit_options={'edit': True}
    )
    draw.add_to(m)

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
# Authentication and request parameters
auth_url = "https://api.hypermeteo.com/auth-b2b/authenticate"
body = {
    "username": "gecosistema",
    "password": "Saferplaces2023!"
}
response = requests.post(auth_url, json=body).json()
token = response['token']
headers = {"Authorization": f"Bearer {token}"}

# Fetch data from API and accumulate rain data
def fetch_acc_rain_data(start_time, end_time):
    current_time = start_time
    accumulated_rain = None
    temp_files = []
    
    while current_time <= end_time:
        subset_time = f'time("{current_time.isoformat(timespec="milliseconds")}Z")'

        params = {
            "request": "GetCoverage",
            "service": "WCS",
            "version": "2.0.0",
            "coverageId": "RADAR_HERA_150M_5MIN__rainrate",
            "format": "application/x-netcdf",
            "subset": [f"Long(12.4,12.9)", f"Lat(43.8,44.2)", subset_time]
        }

        response = requests.get("https://api.hypermeteo.com/b2b-binary/ogc/geoserver/wcs", headers=headers, params=params)
        
        if response.status_code == 200:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.nc')
            tmp_file.write(response.content)
            temp_files.append(tmp_file.name)
            tmp_file.close()

            try:
                ds = xr.open_dataset(tmp_file.name, engine='netcdf4')
                rain = ds['rainrate'].squeeze() if 'rainrate' in ds.variables else None
                if rain is not None:
                    if accumulated_rain is None:
                        accumulated_rain = rain
                    else:
                        accumulated_rain, rain = xr.align(accumulated_rain, rain, join='outer')
                        accumulated_rain = accumulated_rain + rain.fillna(0)
                ds.close()
            except Exception as e:
                st.error(f"Failed to open dataset: {e}")
        else:
            st.error(f"Error fetching data for {current_time}: {response.text}")
            break
        
        current_time += timedelta(minutes=5)
    
    for file_path in temp_files:
        try:
            os.remove(file_path)
        except Exception as e:
            st.error(f"Failed to remove temporary file: {file_path}. Error: {e}")
    
    if accumulated_rain is not None:
        accumulated_rain = accumulated_rain[::-1, :]
    return accumulated_rain

# Convert accumulated rain to GeoTIFF
def convert_accumulated_rain_to_geotiff(accumulated_rain):
    if accumulated_rain is not None:
        rainrate = accumulated_rain.squeeze().values

        if rainrate.ndim > 2:
            rainrate = rainrate[0, :, :]

        lat = accumulated_rain.coords['lat'].values
        lon = accumulated_rain.coords['lon'].values
        lon, lat = np.meshgrid(lon, lat)

        lon_min, lat_max = lon.min(), lat.max()
        cell_size_lon = (lon.max() - lon.min()) / lon.shape[1]
        cell_size_lat = (lat.max() - lat.min()) / lat.shape[0]

        transform = from_origin(lon_min, lat_max, cell_size_lon, abs(cell_size_lat))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
            with rasterio.open(
                tmp_file.name,
                'w',
                driver='GTiff',
                height=rainrate.shape[0],
                width=rainrate.shape[1],
                count=1,
                dtype=rasterio.float32,
                crs='EPSG:4326',
                transform=transform,
            ) as dst:
                dst.write(rainrate.astype(rasterio.float32), 1)
            return tmp_file.name
    return None

# Convert GeoTIFF to Cloud Optimized GeoTIFF (COG)
def convert_to_cog(geotiff_path):
    cog_path = geotiff_path.replace(".tif", "_cog.tif")
    with rasterio.open(geotiff_path) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
            interleave="band",
            dtype=rasterio.float32,
            bigtiff="IF_SAFER",
        )
        with rasterio.open(cog_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                data = src.read(i, resampling=Resampling.nearest)
                dst.write(data, indexes=i)
    return cog_path

# Display COG using Folium
def display_cog_with_folium(cog_path):
    with rasterio.open(cog_path) as src:
        bounds = src.bounds
        band1 = src.read(1)

        vmin = np.min(band1[band1 > 0]) if np.any(band1 > 0) else 0
        vmax = np.max(band1)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap('Blues')
        rgba_image = cmap(norm(band1))
        rgba_image[band1 == 0] = [0, 0, 0, 0]

        m = folium.Map(location=[(bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2], zoom_start=10, tiles="cartodbdark_matter")
        folium.raster_layers.ImageOverlay(
            image=rgba_image,
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            opacity=0.7,
            interactive=True
        ).add_to(m)

        colormap = linear.Blues_09.scale(vmin, vmax)
        colormap.caption = 'Rainfall Intensity'
        colormap.add_to(m)

        folium_static(m)

# Main navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Realtime Pluvial", "Forecast Pluvial", "Coastal Flooding Forecast", "Coastal NowCasting"])

if page == "Realtime Pluvial":
    st.title("Realtime Pluvial Dashboard")
        # Time selection and fetch rain data
    selected_date = st.sidebar.date_input("Select date", datetime.utcnow().date())
    selected_hour = st.sidebar.selectbox("Select hour (UTC)", options=range(24), index=datetime.utcnow().hour)
    selected_minute = st.sidebar.select_slider("Select minute", options=list(range(0, 60, 5)), value=(datetime.utcnow().minute // 5) * 5)
    selected_time = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=selected_hour, minutes=selected_minute)
    cumulative_options = {
        "No Cumulative": timedelta(minutes=5),
        "Last 30 min": timedelta(minutes=30),
        "Last 1 hour": timedelta(hours=1),
        "Last 3 hours": timedelta(hours=3)
    }
    cumulative_interval = st.sidebar.selectbox("Cumulative interval", options=list(cumulative_options.keys()))
    end_time = selected_time
    start_time = end_time - cumulative_options[cumulative_interval]

    # Fetch rain data
    rain_data = fetch_acc_rain_data(start_time, end_time)
    geotiff_path = convert_accumulated_rain_to_geotiff(rain_data)
    if geotiff_path:
        cog_path = convert_to_cog(geotiff_path)
        display_cog_with_folium(cog_path)
        with open(cog_path, "rb") as file:
            st.download_button("Download COG", file, "rainrate_cog.tif", "image/tiff")
    else:
        st.error("Failed to create GeoTIFF.")
    
    scenario = st.selectbox(
        "Select a cumulative rainfall scenario:",
        options=["1h", "3h", "6h", "12h"],
        index=0
    )
    
    radar_intensity = 1.0 + 0.5 * ["1h", "3h", "6h", "12h"].index(scenario)
    flood_intensity = 0.5 + 0.3 * ["1h", "3h", "6h", "12h"].index(scenario)

    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        selected_date = create_date_slider(datetime(2020, 1, 1), datetime(2024, 12, 31))
        point_selection = st.selectbox("Select a point to view time series:", [f"Point {i+1}" for i in range(len(random_points))], index=0)
        selected_point_id = int(point_selection.split()[1])
        
    with col2:
        create_map_viewer_with_barrier("Radar Rainfall Intensity", radar_intensity, "Flood Area", flood_intensity)
        
    with col3:
        create_time_series(selected_date, selected_point_id)

elif page == "Forecast Pluvial":
    st.title("Forecast Pluvial")
    
    scenario = st.selectbox(
        "Select a cumulative rainfall scenario:",
        options=["1h", "3h", "6h", "12h"],
        index=0
    )
    
    radar_intensity = 1.0 + 0.5 * ["1h", "3h", "6h", "12h"].index(scenario)
    flood_intensity = 0.5 + 0.3 * ["1h", "3h", "6h", "12h"].index(scenario)

    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        selected_date = create_date_slider(datetime.now(), datetime.now() + timedelta(hours=48))
        point_selection = st.selectbox("Select a point to view time series:", [f"Point {i+1}" for i in range(len(random_points))], index=0)
        selected_point_id = int(point_selection.split()[1])
        
    with col2:
        create_map_viewer_with_barrier("COSMO Forecast Rainfall", radar_intensity, "Flood Simulation", flood_intensity)
        
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
        create_map_viewer_with_barrier("SEA Model Forecast", radar_intensity=2.0, flood_name="Coastal Flood Simulation", flood_intensity=1.2)
        
    with col3:
        create_time_series(selected_date, selected_point_id)

elif page == "Coastal NowCasting":
    st.title("Coastal NowCasting")
    
    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        selected_date = create_date_slider(datetime.now(), datetime.now() + timedelta(hours=48))
        point_selection = st.selectbox("Select a point to view time series:", [f"Point {i+1}" for i in range(len(random_points))], index=0)
        selected_point_id = int(point_selection.split()[1])
        
    with col2:
        create_map_viewer_with_barrier("SEA Model Forecast", radar_intensity=2.0, flood_name="Coastal Flood Simulation", flood_intensity=1.2)
        
    with col3:
        create_time_series(selected_date, selected_point_id)


