#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[19]:


df = pd.read_csv("insurance_data.csv")
df.head()


# In[20]:


plt.scatter(df.age,df.bought_insurance,marker='.',color='green')


# In[21]:


#split the train and test data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.8)


# In[22]:


X_test


# In[23]:


# import the logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[24]:


model.fit(X_train, y_train)


# In[25]:


X_test


# In[26]:


y_predicted = model.predict(X_test)


# In[27]:


model.predict_proba(X_test)


# In[30]:


model.score(X_test,y_test)


# In[31]:


X_test


# In[32]:


#model.coef_ indicates value of m in y=m*x + b equation
model.coef_


# In[33]:


model.intercept_


# In[34]:


#Lets defined sigmoid function now and do the math with hand
import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))


# In[35]:


def prediction_function(age):
    z = 0.042 * age - 1.53 # 0.04150133 ~ 0.042 and -1.52726963 ~ -1.53
    y = sigmoid(z)
    return y
age = 35
prediction_function(age)


# In[36]:


#0.485 is less than 0.5 which means person with 35 age will not buy insurance
age = 43
prediction_function(age)


# In[37]:


#0.485 is more than 0.5 which means person with 43 will buy the insurance


# In[ ]:





# In[ ]:




