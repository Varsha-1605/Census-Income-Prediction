FEATURE_CONFIGS = {
    'age': {'min': 17, 'max': 90, 'default': 30},
    'fnlwgt': {'min': 10000, 'max': 1500000, 'default': 200000},
    'capital_gain': {'min': 0, 'max': 99999, 'default': 0},
    'capital_loss': {'min': 0, 'max': 4356, 'default': 0},
}

CATEGORICAL_FEATURES = {
    'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                  'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
    'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
                 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
                 '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
    # [Add other categorical features]
}