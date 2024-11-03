# app/Home.py
import streamlit as st

st.set_page_config(
    page_title="Geospatial Data Viewer",
    page_icon="🌍",
    layout="wide"
)

st.title("Welcome to Geospatial Data Viewer")
st.write("""
This application allows you to explore geospatial data through multiple views:
- Temporal analysis with date selection
- Geospatial visualization of raster data
- Time series analysis with interval selection
""")
