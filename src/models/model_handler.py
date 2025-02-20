import pickle
import streamlit as st
import logging
import os

logger = logging.getLogger(__name__)

@st.cache_resource
def load_model():
    model_path = 'model/census_income_model.pkl'
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None
            
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
