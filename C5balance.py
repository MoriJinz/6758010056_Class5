import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'data/tourism_data_cleaned.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

# Separate features and target variable````
X = df.drop(columns=['ProdTaken'])  # Features
y = df['ProdTaken']  # Target variable

# Encode categorical features
categorical_columns = X.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])  # Encode categorical column
    label_encoders[col] = le

# Ensure all numeric columns are integers
X = X.astype(int)

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convert back categorical features from encoded integers to original strings
for col, le in label_encoders.items():
    X_resampled[col] = le.inverse_transform(X_resampled[col])

# Combine the resampled features and target into a new DataFrame
balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
balanced_df['ProdTaken'] = y_resampled

# Save the balanced data to a new CSV file
balanced_file_path = 'data/balanced_tourism_data.csv'  # Set your desired output file path
balanced_df.to_csv(balanced_file_path, index=False)

print(f"Balanced dataset saved to: data/balanced_tourism_data.csv")