import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,roc_curve

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

X=df.drop("churn",axis=1)
y=df["churn"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

randon_forest_model=RandomForestClassifier(random_state=42)
randon_forest_model.fit(X_train,y_train)

y_predict=randon_forest_model.predict(X_test)
y_predict_probabity=randon_forest_model.predict_proba(X_test)[:,1]

print("Classification Report:",classification_report(y_test,y_predict))

cm=confusion_matrix(y_test,y_predict)
plt.figure(figsize=(5,4))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

fpr,tpr,thresholds=roc_curve(y_test,y_predict_probabity)
roc_auc=roc_auc_score(y_test,y_predict_probabity)

plt.figure()
plt.plot(fpr,tpr,color='darkorange',label=f"ROC Curve(AUC={roc_auc:.2f})")
plt.title("Receiver Operating Characteristic")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

