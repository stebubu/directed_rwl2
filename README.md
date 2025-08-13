# directed_rwl2

## Installation

1. Create and activate a Python 3 virtual environment (optional):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Launch the Streamlit application with:

```bash
streamlit run Home.py
```

The dashboard will open in your default web browser.

## Cesium React Web App

The repository also includes a React application that renders a 3D globe using Cesium with city markers and the ability to overlay uploaded GeoTIFF water-depth files.

### Run the React app

1. Navigate to the `webapp` directory and install dependencies:
   ```bash
   cd webapp
   npm install
   ```
2. Start the development server:
   ```bash
   npm run dev
   ```
3. Open the provided URL in your browser. Use the file input to upload a `.tif`/`.tiff` water depth file (COG supported). The imagery will be reprojected to EPSG:4326 and draped over the terrain. Clicking city icons such as Bologna or London will zoom to that location, and Google Photorealistic 3D Tiles buildings will be displayed.
