#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('Churn_Modelling.csv')


# In[3]:


df.shape


# In[6]:


df.info()


# In[4]:


df.describe()


# In[4]:


df.head()


# In[5]:


df.columns


# In[10]:


print(df['Exited'].value_counts())


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[12]:


missing_values=df.isnull().sum()


# In[13]:


print(missing_values)


# In[20]:


sns.countplot(x='Exited',data=df)
plt.title('Churn Distribution')
plt.show()


# In[14]:


correlation_matrix=df.corr()


# In[15]:


sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm')
plt.show()


# In[17]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# In[19]:


features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                     'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                     'EstimatedSalary']


# In[20]:


target= 'Exited'


# In[21]:


X = df[features]
y = df[target]


# In[22]:


categorical_features = ['Geography', 'Gender']
numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                      'HasCrCard', 'IsActiveMember', 'EstimatedSalary']


# In[23]:


preprocessor=ColumnTransformer([('num','passthrough',numerical_features),('cat',OneHotEncoder(),categorical_features)])


# In[24]:


X1=preprocessor.fit_transform(X)


# In[25]:


X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=0.25,random_state=42)


# In[27]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[62]:


dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_predict1=dtc.predict(X_test)

accuracy1=accuracy_score(y_test,y_predict1)
print(accuracy1)

classification_report1=classification_report(y_test,y_predict1)
print(classification_report1)

confusion_matrix1=confusion_matrix(y_test,y_predict1)
print(confusion_matrix1)


# In[64]:


lr=LogisticRegression()
lr.fit(X_train,y_train)
y_predict2=lr.predict(X_test)

accuracy2=accuracy_score(y_test,y_predict2)
print(accuracy2)

classification_report2=classification_report(y_test,y_predict2)
print(classification_report2)

confusion_matrix2=confusion_matrix(y_test,y_predict2)
print(confusion_matrix2)


# In[66]:


gb=GradientBoostingClassifier()
gb.fit(X_train,y_train)
y_predict3=gb.predict(X_test)

accuracy3=accuracy_score(y_test,y_predict3)
print(accuracy3)

classification_report3=classification_report(y_test,y_predict3)
print(classification_report3)

confusion_matrix3=confusion_matrix(y_test,y_predict3)
print(confusion_matrix3)

