import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

