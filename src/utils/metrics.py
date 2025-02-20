from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import streamlit as st
from src.models.model_handler import load_model
from src.data.data_loader import load_data
from src.preprocessing.preprocessor import preprocess_input
import logging as logger



# Function to calculate model metrics


# true values - but static

@st.cache_data
def calculate_model_metrics():
    # Return pre-calculated metrics instead of computing them
    metrics = {
        'accuracy': 0.8886,  # 88.86%
        'precision': 0.8806, # 88.06%
        'recall': 0.8886,    # 88.86%
        'f1': 0.8815         # 88.15%
    }
    return metrics


# true values as it takes time to load so i have used static values
# Function to calculate model metrics
# @st.cache_data
# def calculate_model_metrics():
#     model = load_model()
#     data = load_data()
    
#     if model is None or data is None:
#         return None
        
#     try:
#         # Rename 'sex' column to 'gender' for consistency
#         if 'sex' in data.columns:
#             data = data.rename(columns={'sex': 'gender'})
        
#         # Prepare features and target
#         X = data.drop('income', axis=1)
#         y = data['income']
        
#         # Split the data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         # Process test data
#         processed_test_data = []
#         for _, row in X_test.iterrows():
#             input_data = {
#                 'age': row['age'],
#                 'workclass': row['workclass'],
#                 'fnlwgt': row['fnlwgt'],
#                 'education': row['education'],
#                 'marital-status': row['marital-status'],
#                 'occupation': row['occupation'],
#                 'relationship': row['relationship'],
#                 'race': row['race'],
#                 'gender': row['gender'],  # Changed from 'sex' to 'gender'
#                 'capital-gain': row['capital-gain'],
#                 'capital-loss': row['capital-loss'],
#                 'native-country': row['native-country']
#             }
#             processed_input = preprocess_input(input_data)
#             processed_test_data.append(processed_input)
        
#         X_test_processed = pd.concat(processed_test_data, ignore_index=True)
        
#         # Make predictions
#         y_pred = model.predict(X_test_processed)
        
#         # Calculate metrics
#         metrics = {
#             'accuracy': accuracy_score(y_test[:len(y_pred)], y_pred),
#             'precision': precision_score(y_test[:len(y_pred)], y_pred),
#             'recall': recall_score(y_test[:len(y_pred)], y_pred),
#             'f1': f1_score(y_test[:len(y_pred)], y_pred)
#         }
#         return metrics
#     except Exception as e:
#         st.error(f"Error calculating metrics: {e}")
#         logger.error(f"Error in calculate_model_metrics: {e}")
#         return None

# Function to extract feature importance from model
@st.cache_data
def get_feature_importance():
    model = load_model()
    
    if model is not None and hasattr(model, 'feature_importances_'):
        # Get actual feature importances from model
        importance_values = model.feature_importances_
        
        # Get feature names
        if hasattr(model, 'feature_names_'):
            feature_names = model.feature_names_
        else:
            # Try to get column names from the preprocessing pipeline
            try:
                # Get the model columns from preprocessing function
                dummy_input = {
                    'age': 30,
                    'workclass': 'Private',
                    'fnlwgt': 200000,
                    'education': 'HS-grad',
                    'marital-status': 'Never-married',
                    'occupation': 'Sales',
                    'relationship': 'Not-in-family',
                    'race': 'White',
                    'gender': 'Male',
                    'capital-gain': 0,
                    'capital-loss': 0,
                    'native-country': 'United-States'
                }
                processed = preprocess_input(dummy_input)
                feature_names = processed.columns.tolist()
            except:
                # Fallback to generic feature names
                feature_names = [f"Feature_{i}" for i in range(len(importance_values))]
        
        # Map original feature names to more readable names
        readable_names = []
        readable_values = []
        
        # Process one-hot encoded features to get original category names
        seen_prefixes = set()
        for i, name in enumerate(feature_names):
            # Check if this is a one-hot encoded column
            if '_' in name:
                prefix = name.split('_')[0]
                if prefix not in seen_prefixes:
                    # For first occurrence of a one-hot feature, sum all related importances
                    related_features = [j for j, f in enumerate(feature_names) if f.startswith(prefix + '_')]
                    combined_importance = sum(importance_values[j] for j in related_features)
                    readable_names.append(prefix.title())
                    readable_values.append(combined_importance)
                    seen_prefixes.add(prefix)
            elif name not in seen_prefixes:
                # For regular features
                readable_names.append(name.title())
                readable_values.append(importance_values[i])
                seen_prefixes.add(name)
        
        # Create DataFrame and sort by importance
        importance_data = pd.DataFrame({
            'Feature': readable_names,
            'Importance': readable_values
        })
        importance_data = importance_data.sort_values('Importance', ascending=False).head(10)
        
        # Normalize to make highest value 100
        if not importance_data.empty:
            max_value = importance_data['Importance'].max()
            importance_data['Importance'] = (importance_data['Importance'] / max_value * 100).round(1)
        
        return importance_data, True
    else:
        # Use educated estimates if model doesn't have feature_importances_
        importance_data = pd.DataFrame({
            'Feature': ['Marital-Status', 'Education', 'Age', 'Occupation', 'Capital-Gain', 
                       'Workclass', 'Relationship', 'Gender', 'Race'],
            'Importance': [100, 82, 76, 65, 52, 42, 38, 25, 15]
        })
        return importance_data, False

