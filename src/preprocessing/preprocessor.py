import pandas as pd

# Function to preprocess input data
def preprocess_input(input_data):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Handle education (label encoded in your original code)
    education_mapping = {
        'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4,
        '10th': 5, '11th': 6, '12th': 7, 'HS-grad': 8, 'Some-college': 9,
        'Assoc-voc': 10, 'Assoc-acdm': 11, 'Bachelors': 12, 'Masters': 13,
        'Prof-school': 14, 'Doctorate': 15
    }
    input_df['education'] = input_df['education'].map(education_mapping)
    
    # Handle gender (binary encoded)
    input_df['gender'] = input_df['gender'].map({'Male': 1, 'Female': 0})
    
    # One-hot encode categorical variables
    # Workclass
    workclass_cols = ['workclass_Federal-gov', 'workclass_Local-gov', 'workclass_Never-worked',
                      'workclass_Private', 'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc',
                      'workclass_State-gov', 'workclass_Without-pay']
    
    for col in workclass_cols:
        input_df[col] = 0
    
    workclass_value = input_df['workclass'].iloc[0]
    col_name = f'workclass_{workclass_value}'
    if col_name in workclass_cols:
        input_df[col_name] = 1
    
    # Marital status
    marital_cols = ['marital-status_Divorced', 'marital-status_Married-AF-spouse',
                    'marital-status_Married-civ-spouse', 'marital-status_Married-spouse-absent',
                    'marital-status_Never-married', 'marital-status_Separated',
                    'marital-status_Widowed']
    
    for col in marital_cols:
        input_df[col] = 0
    
    marital_value = input_df['marital-status'].iloc[0]
    col_name = f'marital-status_{marital_value}'
    if col_name in marital_cols:
        input_df[col_name] = 1
    
    # Occupation
    occupation_cols = ['occupation_Adm-clerical', 'occupation_Armed-Forces',
                       'occupation_Craft-repair', 'occupation_Exec-managerial',
                       'occupation_Farming-fishing', 'occupation_Handlers-cleaners',
                       'occupation_Machine-op-inspct', 'occupation_Other-service',
                       'occupation_Priv-house-serv', 'occupation_Prof-specialty',
                       'occupation_Protective-serv', 'occupation_Sales',
                       'occupation_Tech-support', 'occupation_Transport-moving']
    
    for col in occupation_cols:
        input_df[col] = 0
    
    occupation_value = input_df['occupation'].iloc[0]
    col_name = f'occupation_{occupation_value}'
    if col_name in occupation_cols:
        input_df[col_name] = 1
    
    # Relationship
    relationship_cols = ['relationship_Husband', 'relationship_Not-in-family',
                         'relationship_Other-relative', 'relationship_Own-child',
                         'relationship_Unmarried', 'relationship_Wife']
    
    for col in relationship_cols:
        input_df[col] = 0
    
    relationship_value = input_df['relationship'].iloc[0]
    col_name = f'relationship_{relationship_value}'
    if col_name in relationship_cols:
        input_df[col_name] = 1
    
    # Race
    race_cols = ['race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander',
                'race_Black', 'race_Other', 'race_White']
    
    for col in race_cols:
        input_df[col] = 0
    
    race_value = input_df['race'].iloc[0]
    col_name = f'race_{race_value}'
    if col_name in race_cols:
        input_df[col_name] = 1
    
    # Handle native-country (frequency encoded)
    # Simplified frequency encoding - in production, use the exact same mapping from training
    us_countries = ['United-States']
    if input_df['native-country'].iloc[0] in us_countries:
        input_df['native-country'] = 0.91  # Approximate frequency for US
    else:
        input_df['native-country'] = 0.03  # Approximate frequency for other countries
    
    # Drop original categorical columns that have been encoded
    input_df.drop(['workclass', 'marital-status', 'occupation', 'relationship', 'race'], axis=1, inplace=True)
    
    # Ensure we have all required columns for the model
    model_columns = ['age', 'fnlwgt', 'education', 'gender', 'capital-gain', 'capital-loss',
                     'native-country', 'workclass_Federal-gov', 'workclass_Local-gov',
                     'workclass_Never-worked', 'workclass_Private', 'workclass_Self-emp-inc',
                     'workclass_Self-emp-not-inc', 'workclass_State-gov', 'workclass_Without-pay',
                     'marital-status_Divorced', 'marital-status_Married-AF-spouse',
                     'marital-status_Married-civ-spouse', 'marital-status_Married-spouse-absent',
                     'marital-status_Never-married', 'marital-status_Separated',
                     'marital-status_Widowed', 'occupation_Adm-clerical', 'occupation_Armed-Forces',
                     'occupation_Craft-repair', 'occupation_Exec-managerial',
                     'occupation_Farming-fishing', 'occupation_Handlers-cleaners',
                     'occupation_Machine-op-inspct', 'occupation_Other-service',
                     'occupation_Priv-house-serv', 'occupation_Prof-specialty',
                     'occupation_Protective-serv', 'occupation_Sales', 'occupation_Tech-support',
                     'occupation_Transport-moving', 'relationship_Husband',
                     'relationship_Not-in-family', 'relationship_Other-relative',
                     'relationship_Own-child', 'relationship_Unmarried', 'relationship_Wife',
                     'race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black',
                     'race_Other', 'race_White']
    
    # Filter columns to match model input requirements
    model_input = input_df[input_df.columns.intersection(model_columns)]
    
    # Add any missing columns with zeros
    for col in model_columns:
        if col not in model_input.columns:
            model_input[col] = 0
    
    # Reorder columns to match model expectations
    model_input = model_input[model_columns]
    
    return model_input


