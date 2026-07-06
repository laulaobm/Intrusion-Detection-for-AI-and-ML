import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
#No idea why but it seems the csv files are named incorrectly...
train_df = pd.read_csv('../../data/UNSW_NB15_testing-set.csv')
test_df = pd.read_csv('../../data/UNSW_NB15_training-set.csv')

train_df = train_df.drop(columns=['id'])
test_df = test_df.drop(columns=['id'])

X_train = train_df.drop(columns=['attack_cat', 'label'])
y_train_binary = train_df['label']
y_train_multi = train_df['attack_cat']

X_test = test_df.drop(columns=['attack_cat', 'label'])
y_test_binary = test_df['label']
y_test_multi = test_df['attack_cat']

categorical_cols = ['proto', 'service', 'state']
X_train = pd.get_dummies(X_train, columns=categorical_cols)
X_test = pd.get_dummies(X_test, columns=categorical_cols)

X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

label_encoder = LabelEncoder()
y_train_multi_encoded = label_encoder.fit_transform(y_train_multi)
y_test_multi_encoded = label_encoder.transform(y_test_multi)

scaler = MinMaxScaler()
numeric_cols = X_train.select_dtypes(include=['int64', 'float64', 'int32', 'float32', 'uint8']).columns

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

print(X_train.shape)
print(X_test.shape)

smote = SMOTE(random_state=42)

X_train_multi_resampled, y_train_multi_resampled = smote.fit_resample(X_train, y_train_multi_encoded)

print(X_train.shape)
print(X_train_multi_resampled.shape)

X_train_binary_resampled, y_train_binary_resampled = smote.fit_resample(X_train, y_train_binary)

joblib.dump(scaler, 'min_max_scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

X_train_multi_resampled.to_parquet('X_train_multi_resampled.parquet')
X_train_binary_resampled.to_parquet('X_train_binary_resampled.parquet')
X_test.to_parquet('X_test.parquet')

pd.Series(y_train_multi_resampled, name='target').to_frame().to_parquet('../../data/y_train_multi_resampled.parquet')
pd.Series(y_train_binary_resampled, name='target').to_frame().to_parquet('../../data/y_train_binary_resampled.parquet')

pd.Series(y_test_multi_encoded, name='target').to_frame().to_parquet('../../data/y_test_multi.parquet')
pd.Series(y_test_binary, name='target').to_frame().to_parquet('../../data/y_test_binary.parquet')

#prints done by gemini to examine data

# Displays the first 5 rows to verify that categorical columns are now one-hot encoded
# and numerical columns are scaled.
print("--- First 5 Rows of Resampled Data ---")
print(X_train_multi_resampled.head())

# Pulls 5 random rows to give a broader view of the data structure beyond the top rows.
print("\n--- 5 Random Rows of Resampled Data ---")
print(X_train_multi_resampled.sample(5))

# Compares the row counts to show exactly how many synthetic records SMOTE added.
print("\n--- Dataset Shapes ---")
print(f"Resampled Training Data Shape: {X_train_multi_resampled.shape}")
print(f"Test Data Shape: {X_test.shape}")

# Confirms that all 9 specific attack categories now have the exact same number of samples.
print("\n--- Multi-Class Balance (Attack Categories) ---")
print(pd.Series(y_train_multi_resampled).value_counts())

# Confirms that the generic 'Normal' vs 'Attack' labels are now perfectly balanced (50/50).
print("\n--- Binary Class Balance (Normal vs. Attack) ---")
print(pd.Series(y_train_binary_resampled).value_counts())

# Generates a statistical summary. You should check that the 'min' row is 0.0
# and the 'max' row is 1.0 for all features, proving Min-Max scaling worked.
print("\n--- Min-Max Scaling Verification ---")
print(X_train_multi_resampled.describe())