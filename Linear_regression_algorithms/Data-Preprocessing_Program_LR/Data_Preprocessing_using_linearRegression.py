#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n


# In[13]:


df=pd.read_csv("hiring.csv")
df


# In[14]:


df.experience = df.experience.fillna("zero")
df


# In[15]:


df.experience = df.experience.apply(w2n.word_to_num)
df


# In[17]:


import math
median_test_score = math.floor(df['test_score(out of 10)'].mean())
median_test_score


# In[18]:


df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_test_score)
df


# In[20]:


reg = linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])


# In[21]:


reg.predict([[2,9,6]])



# In[22]:


reg.predict([[12,10,10]])


# In[ ]:




