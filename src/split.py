import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/labels.csv')
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# 저장
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)
