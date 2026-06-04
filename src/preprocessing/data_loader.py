import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

train_df = pd.read_csv('../../data/UNSW_NB15_training-set.csv')
test_df = pd.read_csv('../../data/UNSW_NB15_testing-set.csv')

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

pd.Series(y_train_multi_resampled, name='target').to_frame().to_parquet('y_train_multi_resampled.parquet')
pd.Series(y_train_binary_resampled, name='target').to_frame().to_parquet('y_train_binary_resampled.parquet')

pd.Series(y_test_multi_encoded, name='target').to_frame().to_parquet('y_test_multi.parquet')
pd.Series(y_test_binary, name='target').to_frame().to_parquet('y_test_binary.parquet')