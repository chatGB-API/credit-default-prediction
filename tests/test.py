from src.preprocess import load_and_clean, engineer_features, clean_column_names
import joblib,json
import pandas as pd

df = load_and_clean('data/raw/cs-test.csv')

if 'Id' in df.columns:
    ids = df['Id'].copy()
elif 'id' in df.columns:
    ids = df['id'].copy()
else:
    ids = pd.Series(range(1, len(df) + 1), name='Id')

df = engineer_features(df)
df = clean_column_names(df)

X = df.drop('SeriousDlqin2yrs', axis=1)

model = joblib.load('models/lgbm_model.pkl')
thr = joblib.load('models/best_thr.pkl')
pred = model.predict_proba(X)[:,1]

df['SeriousDlqin2yrs'] = (pred>float(thr)).astype(int)

submission = pd.DataFrame({'Id': ids.values, 'Survived': df['SeriousDlqin2yrs'].values})
submission.to_csv('submission.csv', index=False)

print(submission.head())

import os

submission_path = os.path.abspath('submission.csv')
print("保存路径:", submission_path)
print("文件是否存在:", os.path.exists(submission_path))




