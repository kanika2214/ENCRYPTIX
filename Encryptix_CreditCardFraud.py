#!/usr/bin/env python
# coding: utf-8

# # Data Loading and initial imports and exploration

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


#data import
df=pd.read_csv('creditcard.csv',encoding='latin1')


# # Exploratory Data Analysis
# 

# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


#no null values


# In[10]:


print(df['Class'].value_counts())


# In[11]:


#clearly fraud values are very less in number than real values


# In[12]:


#countplot
plt.figure(figsize=(6,4))
sns.countplot(x='Class',data=df)
plt.title('Class Distribution')
plt.show()


# In[54]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Class',y='Amount',data=df,palette=['blue','red'])


# In[44]:


#correlation matrix
plt.figure(figsize=(12,10))
corr=df.corr()
sns.heatmap(corr,cmap='coolwarm',annot=False)
plt.title('Correlation Matrix')
plt.show()


# In[45]:


fraud_df=df[df['Class']==1]
normal_df=df[df['Class']==0]


# In[46]:


#histogram depicting the frequency of transactions in different amount intervals. here is is visible that the maximum thansations happened for very small values.
plt.figure(figsize=(10,4))
sns.histplot(fraud_df['Amount'],bins=50,kde=True)
plt.title('Transaction amount Distribution')
plt.show()


# In[49]:


#histogram depicting the frequency of transactions in different amount intervals. here is is visible that the maximum thansations happened for very small values.
plt.figure(figsize=(10,4))
sns.histplot(normal_df['Amount'],bins=50,kde=True)
plt.title('Transaction amount Distribution')
plt.show()


# In[56]:


fraud_df['Amount'].describe()


# In[55]:


normal_df['Amount'].describe()


# # Data preprocessing

# In[15]:


#importing new libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# In[16]:


X=df.drop('Class',axis=1)  #features
y=df['Class'] #target variable


# In[17]:


scaler=StandardScaler()


# In[18]:


X_scaled=scaler.fit_transform(X)


# In[19]:


X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42,stratify=y)


# In[20]:


#balancing the data
smote=SMOTE(random_state=42)


# In[21]:


X_train_balanced,y_train_balanced=smote.fit_resample(X_train,y_train)


# In[22]:


#original data counts
print(pd.Series(y_train).value_counts())


# In[23]:


#new data counts, balanced data
print(pd.Series(y_train_balanced).value_counts())


# # Training and Evaluation using Logistic regression

# In[24]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[25]:


#modelling
logisticregression=LogisticRegression()
logisticregression.fit(X_train_balanced,y_train_balanced)
y_predict1=logisticregression.predict(X_test)


# In[26]:


#evaluations
accuracy_score(y_test,y_predict1)


# In[27]:


confusion_matrix(y_test,y_predict1)


# In[28]:


classification_report(y_test,y_predict1)


# # Training and evaluation using Decision Trees

# In[29]:


from sklearn.tree import DecisionTreeClassifier 


# In[30]:


Dec_Tree=DecisionTreeClassifier(random_state=42)


# In[31]:


Dec_Tree.fit(X_train_balanced,y_train_balanced)


# In[32]:


y_predict2=Dec_Tree.predict(X_test)


# In[33]:


#evaluations
accuracy_score(y_test,y_predict2)


# In[34]:


confusion_matrix(y_test,y_predict2)


# In[35]:


classification_report(y_test,y_predict2)


# # Training and evaulation using Random Forests

# In[36]:


from sklearn.ensemble import RandomForestClassifier


# In[37]:


ran_for=RandomForestClassifier(random_state=42)


# In[38]:


ran_for.fit(X_train_balanced,y_train_balanced)


# In[39]:


y_predict3=ran_for.predict(X_test)


# In[40]:


#evaluations
accuracy_score(y_test,y_predict3)


# In[41]:


confusion_matrix(y_test,y_predict3)


# In[42]:


classification_report(y_test,y_predict3)


# In[ ]:


#Logistic trees are least accurate amongst the three while Random forests are most accurate.
#Logistic regression assumes linear relationship between features adn target which might not be the case always. 
#Decision trees are prone to overfitting if the trees get complex.
#Random forests are an ensemble of trees which ensure that overfitting is reduced due to taking average and it is good in identifiying general trends in data and hence it has the highest accuracy score.

