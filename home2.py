# app/Home.py
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timezone, timedelta
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
import matplotlib.pyplot as plt
import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import hashlib
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


# AWS Credentials Input
st.sidebar.title("AWS Credentials")
aws_access_key_id = st.sidebar.text_input("AWS Access Key ID", type="password")
aws_secret_access_key = st.sidebar.text_input("AWS Secret Access Key", type="password")
aws_region = st.sidebar.text_input("AWS Region", value="us-east-1")

# Function to upload a file to S3
def s3_upload(local_file, s3_file, bucket_name="s3-directed"):
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    s3 = session.client("s3")
    try:
        s3.upload_file(local_file, bucket_name, s3_file)
        st.success(f"Uploaded {local_file} to {s3_file}")
    except Exception as e:
        st.error(f"Failed to upload to S3: {e}")

# New function to generate rainfall map based on selected modality
def generate_rainfall_map(modality, start_time, end_time):
    st.write(f"Generating rainfall intensity map using {modality} method...")

    if modality == "HERA":
        # Fetch HERA radar data
        accumulated_rain = fetch_acc_rain_data(start_time, end_time)
        if accumulated_rain is None:
            st.error("No data retrieved from HERA radar.")
            return None

        # Process accumulated rain to GeoTIFF
        geotiff_path = convert_accumulated_rain_to_geotiff(accumulated_rain)
        if geotiff_path:
            st.success(f"HERA rainfall intensity map generated: {geotiff_path}")
            return geotiff_path
        else:
            st.error("Failed to generate GeoTIFF from HERA data.")
            return None
    else:
        # Simulate data generation for other modalities
        data = np.random.random((100, 100)) * 10  # Simulated rainfall intensity in mm
        lat_min, lat_max = 43.8, 44.2
        lon_min, lon_max = 12.4, 12.9
        cell_size_lon = (lon_max - lon_min) / data.shape[1]
        cell_size_lat = (lat_max - lat_min) / data.shape[0]

        # Create a temporary GeoTIFF file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_file:
            transform = from_origin(lon_min, lat_max, cell_size_lon, abs(cell_size_lat))
            with rasterio.open(
                tmp_file.name,
                'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=rasterio.float32,
                crs='EPSG:4326',
                transform=transform
            ) as dst:
                dst.write(data.astype(rasterio.float32), 1)
            geotiff_path = tmp_file.name
            st.success(f"Rainfall intensity map generated for {modality}: {geotiff_path}")
            return geotiff_path
# Function to display the generated rainfall map using Folium
def display_rainfall_map(geotiff_path):
    try:
        with rasterio.open(geotiff_path) as src:
            bounds = src.bounds
            rainfall_data = src.read(1)
            vmin, vmax = rainfall_data.min(), rainfall_data.max()

            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = colormaps.get_cmap('Blues')
            rgba_image = cmap(norm(rainfall_data))
            rgba_image[rainfall_data == 0] = [0, 0, 0, 0]

            m = folium.Map(location=[(bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2], zoom_start=10)
            folium.raster_layers.ImageOverlay(
                image=rgba_image,
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                opacity=0.7
            ).add_to(m)

            colormap = linear.Blues_09.scale(vmin, vmax)
            colormap.caption = 'Rainfall Intensity (mm)'
            colormap.add_to(m)

            st_folium(m)
    except Exception as e:
        st.error(f"Error displaying rainfall map: {e}")




# Flood map generation function
def generate_flood_map(geotiff_path):
    # Define S3 paths
    s3_file = "Directed/Rimini/rain_radar_hera.tif"
    s3_bucket = "s3-directed"
    s3_path = f"s3://{s3_bucket}/{s3_file}"

    # Upload the file to S3
    s3_upload(geotiff_path, s3_file)

    # Define process request parameters
    url = "http://pygeoapi-saferplaces-lb-409838694.us-east-1.elb.amazonaws.com/processes/safer-rain-process/execution"
    dem_file = "s3://saferplaces.co/Directed/Rimini/dtm_cropped_32633.bld.tif"
    output_file = "s3://saferplaces.co/Directed/Rimini/wd_rain_radar_hera.tif"
    rain_file = s3_path

    body = {
        "inputs": {
            "dem": dem_file,
            "rain": rain_file,
            "water": output_file,
            "overwrite": True,
            "sync": True,
            "debug": True,
        }
    }

    # Make the API request
    response = requests.post(url, json=body)
    if response.status_code == 200:
        st.write("Flood map generated successfully.")
    else:
        st.error(f"Flood map generation failed: {response.text}")
        return None

    # Download the generated file from S3
    download_url = output_file.replace("s3://", "https://s3.amazonaws.com/")
    output_local_path = os.path.basename(download_url)

    try:
        response = requests.get(download_url)
        if response.status_code == 200:
            with open(output_local_path, "wb") as f:
                f.write(response.content)
            st.success("Downloaded flood map successfully.")
            return output_local_path
        else:
            st.error(f"Failed to download flood map: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error downloading file: {e}")
        return None

# Function to display the flood map with Folium
def display_flood_map(flood_map_path):
    try:
        with rasterio.open(flood_map_path) as src:
            bounds = src.bounds
            flood_data = src.read(1)
            vmin, vmax = flood_data.min(), flood_data.max()

            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = colormaps.get_cmap('Blues')
            rgba_image = cmap(norm(flood_data))
            rgba_image[flood_data == 0] = [0, 0, 0, 0]

            m = folium.Map(location=[(bounds.bottom + bounds.top) / 2, (bounds.left + bounds.right) / 2], zoom_start=10)
            folium.raster_layers.ImageOverlay(
                image=rgba_image,
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                opacity=0.7
            ).add_to(m)

            colormap = linear.Blues_09.scale(vmin, vmax)
            colormap.caption = 'Flood Depth'
            colormap.add_to(m)

            st_folium(m)
    except Exception as e:
        st.error(f"Error displaying flood map: {e}")



# Function to create a date slider with specific ranges
def create_date_slider(start, end):
    dates = pd.date_range(start=start, end=end, freq='h')
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

    # Add flood overlay with specified intensity
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
    st_folium(m)

# Function to create time series plot for a specific point
'''def create_time_series(selected_date, point_id):
    dates = pd.date_range(start=selected_date - timedelta(days=2), end=selected_date, freq='h')
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

    st.plotly_chart(fig, use_container_width=True)'''


def create_time_series(selected_date, point_id):
    # Generate data for the selected date with 15-minute intervals
    dates = pd.date_range(start=selected_date, end=selected_date + timedelta(days=1), freq='15T')
    values = np.random.uniform(0, 10, size=len(dates))  # Random rainfall intensity in mm
    df = pd.DataFrame({
        'Date': dates,
        'Rainfall Intensity (mm)': values
    })

    # Calculate cumulative rainfall
    df['Cumulative Rainfall (mm)'] = df['Rainfall Intensity (mm)'].cumsum()

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot histogram (bar chart) on the primary y-axis
    ax1.bar(df['Date'], df['Rainfall Intensity (mm)'], width=0.01, color='blue', label='Rainfall Intensity (mm)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Rainfall Intensity (mm)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Format x-axis for better readability
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=1))
    plt.xticks(rotation=45)

    # Plot cumulative rainfall on the secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(df['Date'], df['Cumulative Rainfall (mm)'], color='red', label='Cumulative Rainfall (mm)', linewidth=2)
    ax2.set_ylabel('Cumulative Rainfall (mm)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add a legend
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)

    # Set title
    plt.title(f"Rainfall Intensity and Cumulative Rainfall for Point {point_id}")

    # Show plot in Streamlit
    st.pyplot(fig)



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
    try:
        import matplotlib
        if hasattr(matplotlib, 'colormaps'):  # For matplotlib >= 3.7.0
            cmap = matplotlib.colormaps['Blues']
        else:  # For older versions
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap('Blues')

        with rasterio.open(cog_path) as src:
            bounds = src.bounds
            band1 = src.read(1)

            vmin = np.min(band1[band1 > 0]) if np.any(band1 > 0) else 0
            vmax = np.max(band1)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
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

            st_folium(m)
    except Exception as e:
        st.error(f"Error displaying COG with Folium: {e}")

# Main navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Realtime Pluvial", "Forecast Pluvial", "Coastal Flooding Forecast", "Coastal NowCasting"])

if page == "Realtime Pluvial":
    st.title("Realtime Pluvial Dashboard")

    # Initialize geotiff_path
    geotiff_path = None
        # Time selection and fetch rain data
    selected_date = st.sidebar.date_input("Select date", datetime.now(timezone.utc).date())
    selected_hour = st.sidebar.selectbox("Select hour (UTC)", options=range(24), index=datetime.now(timezone.utc).hour)
    selected_minute = st.sidebar.select_slider("Select minute", options=list(range(0, 60, 5)), value=(datetime.now(timezone.utc).minute // 5) * 5)
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

    # Rainfall generation modality selection
    st.sidebar.subheader("Rainfall Intensity Generation Method")
    modalities = ["HERA", "ARPAE", "INTERP"]
    selected_modality = st.sidebar.radio("Select modality:", modalities)

    # Button to generate rainfall map
   # Button to generate rainfall map
# Button to generate rainfall map
    if st.sidebar.button("Generate Rainfall Map"):
        # Generate rainfall map and store in session state
        geotiff_path = generate_rainfall_map(selected_modality, start_time, end_time)
        if geotiff_path:
            st.session_state.geotiff_path = geotiff_path  # Store in session state
            st.success("Rainfall map generated successfully!")
        else:
            st.error("Failed to generate GeoTIFF.")
    
    # Check if a rainfall map already exists in session state
    if "geotiff_path" in st.session_state and st.session_state.geotiff_path:
        # Display the rainfall map
        display_rainfall_map(st.session_state.geotiff_path)
    
        # Convert to COG format
        cog_path = convert_to_cog(st.session_state.geotiff_path)
        if cog_path:
            # Display the COG map
            display_cog_with_folium(cog_path)
    
            # Add a download button for the COG file
            with open(cog_path, "rb") as file:
                st.download_button("Download COG", file, "rainrate_cog.tif", "image/tiff")
        else:
            st.error("Failed to create COG.")
    else:
        st.info("Click 'Generate Rainfall Map' to create a map.")
    # Generate flood map button
    if st.button("Generate Flood Map"):
        #geotiff_path = "/path/to/local/geotiff.tif"  # Replace with actual path or use generated geotiff from previous steps
        flood_map_path = generate_flood_map(geotiff_path)
        if flood_map_path:
            display_flood_map(flood_map_path)
    
'''    scenario = st.selectbox(
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
        if st.button("Generate Map with Overlays"):
            create_map_viewer_with_barrier(
                radar_name="Radar Rainfall Intensity",
                radar_intensity=50,
                flood_name="Flood Area",
                flood_intensity=30
            )
        #create_map_viewer_with_barrier("Radar Rainfall Intensity", radar_intensity, "Flood Area", flood_intensity)
        
    with col3:
        create_time_series(selected_date, selected_point_id)'''

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
        # Button to trigger map creation

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

