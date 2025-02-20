import pandas as pd
import streamlit as st
import logging
import os

logger = logging.getLogger(__name__)

@st.cache_data
def load_data():
    data_path = 'dataset/adult.csv'
    try:
        if not os.path.exists(data_path):
            st.error(f"Data file not found at {data_path}")
            return None
            
        data = pd.read_csv(data_path)
        
        if 'sex' in data.columns:
            data = data.rename(columns={'sex': 'gender'})
        
        data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        logger.error(f"Error in load_data: {e}")
        return None

@st.cache_data
def load_sample_data():
    data = load_data()
    return data.sample(min(1000, len(data))) if data is not None else None
