"""App configuration for CSS and page layout."""
#imports
import streamlit as st
#Function to set up the page configuration
def setup_page_config():
    st.set_page_config(
        layout="wide",
        page_title="Re-Ranking App",
        page_icon="components/main/fav.ico"
    )
    st.markdown(
        """
        <style>
        span[data-baseweb="tag"] {
            background-color: #136eaa !important;
        </style>
        """,
        unsafe_allow_html=True
    )