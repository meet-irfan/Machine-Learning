#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("HR_comma_sep.csv")
df.head()


# In[4]:


#Data exploration and visualization
left = df[df.left==1]
left.shape


# In[5]:


retained = df[df.left==0]
retained.shape


# In[7]:


#Average numbers for all columns
df.groupby('left').mean()


# In[ ]:


#From above table we can draw following conclusions,

**Satisfaction Level**: Satisfaction level seems to be relatively low (0.44) in employees leaving the firm vs the retained ones (0.66)
**Average Monthly Hours**: Average monthly hours are higher in employees leaving the firm (199 vs 207)
**Promotion Last 5 Years**: Employees who are given promotion are likely to be retained at firm
    
    


# In[8]:


pd.crosstab(df.salary,df.left).plot(kind='bar')


# In[9]:


#Above bar chart shows employees with high salaries are likely to not leave the company

#Department wise employee retention rate
pd.crosstab(df.Department,df.left).plot(kind='bar')


# In[ ]:


#From above chart there seem to be some impact of department on employee retention but it is not major hence we will ignore department in our analysis

From the data analysis so far we can conclude that we will use following variables as independant variables in our model
**Satisfaction Level**
**Average Monthly Hours**
**Promotion Last 5 Years**
**Salary**


# In[10]:


subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
subdf.head()


# In[ ]:


#Tackle salary dummy variable

Salary has all text data. It needs to be converted to numbers and we will use dummy variable for that. Check my one hot encoding tutorial to understand purpose behind dummy variables.


# In[11]:


salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
df_with_dummies = pd.concat([subdf,salary_dummies],axis='columns')
df_with_dummies.head()


# In[12]:


#Now we need to remove salary column which is text data. It is already replaced by dummy variables so we can safely remove it


# In[13]:


df_with_dummies.drop('salary',axis='columns',inplace=True)
df_with_dummies.head()


# In[14]:


X = df_with_dummies
X.head()


# In[15]:


y = df.left


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)


# In[17]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[18]:


model.fit(X_train, y_train)


# In[19]:


model.predict(X_test)


# In[20]:


model.score(X_test,y_test)


# In[ ]:




