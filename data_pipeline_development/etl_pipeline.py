import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np

# Load data from CSV
csv_path = 'project_1/fake_job_postings.csv'
df = pd.read_csv(csv_path)
print('Original Data Shape:', df.shape)
print('Columns:', df.columns.tolist())
print(df.head())

# Select columns for processing
numeric_features = ['telecommuting', 'has_company_logo', 'has_questions']
categorical_features = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']

# Preprocessing pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Fit and transform the data
X = df[numeric_features + categorical_features]
X_processed = preprocessor.fit_transform(X)

# Get feature names
cat_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
all_features = numeric_features + list(cat_feature_names)
df_processed = pd.DataFrame(X_processed, columns=np.array(all_features))

print('\nProcessed Data Shape:', df_processed.shape)
print(df_processed.head())

# Save processed data
output_path = 'project_1/processed_fake_job_postings.csv'
df_processed.to_csv(output_path, index=False)
print(f'Processed data saved to {output_path}') 