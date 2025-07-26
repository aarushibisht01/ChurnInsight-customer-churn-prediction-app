import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

file='data/real_world_dataset.csv'
df=pd.read_csv(file)

info={
    "shape":df.shape,
    "data_types":df.dtypes.to_dict(),
    "missing_values":df.isnull().sum().to_dict(),
    "duplicates":df.duplicated().sum()
}

obj_col=df.select_dtypes(include='object').columns.to_list()

for col in obj_col:
    try:
        df[col]=pd.to_numeric(df[col],errors='coerce')
    except:
        continue

obj_col=df.select_dtypes(include='object').columns.to_list()

threshold=0.5*len(df)
df=df.loc[:,df.isnull().sum()<threshold]

for col in df.columns:
    if df[col].dtype=='object':
        df[col].fillna(df[col].mode()[0],inplace=True)
    else:
        df[col].fillna(df[col].mean(),inplace=True)

df.drop_duplicates(inplace=True)

label_encoder=LabelEncoder()

for col in obj_col:
    if col in df.columns:
        df[col]=label_encoder.fit_transform(df[col])
        
clean_dataset='data/clean_real_world_dataset.csv'
df.to_csv(clean_dataset,index=False)

sns.set(style='whitegrid')
plt.figure(figsize=(10,6))

col=df.columns[-1]
sns.countplot(data=df,x=col,palette='Set2')
plt.title(f"Distribution of target column: {col}")

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm",fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

numeric_columns=df.select_dtypes(include='number').columns[:5]
sns.pairplot(df[numeric_columns],corner=True)
plt.suptitle('Pairplot of selected features',y=1.02)
plt.show()
