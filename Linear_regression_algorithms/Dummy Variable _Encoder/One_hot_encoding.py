#!/usr/bin/env python
# coding: utf-8

# <h2>Categorical Variables and One Hot Encoding</h2>

# In[8]:


import pandas as pd


# In[9]:


df = pd.read_csv("homeprices.csv")
df


# <h2 style='color:purple'>Using pandas to create dummy variables</h2>

# In[10]:


dummies = pd.get_dummies(df.town)
dummies


# In[11]:


merged = pd.concat([df,dummies],axis='columns')
merged


# In[12]:


final = merged.drop(['town'], axis='columns')
final


# <h3 style='color:purple'>Dummy Variable Trap</h3>

# When you can derive one variable from other variables, they are known to be multi-colinear. Here
# if you know values of california and georgia then you can easily infer value of new jersey state, i.e. 
# california=0 and georgia=0. There for these state variables are called to be multi-colinear. In this
# situation linear regression won't work as expected. Hence you need to drop one column. 

# **NOTE: sklearn library takes care of dummy variable trap hence even if you don't drop one of the 
#     state columns it is going to work, however we should make a habit of taking care of dummy variable
#     trap ourselves just in case library that you are using is not handling this for you**

# In[13]:


final = final.drop(['west windsor'], axis='columns')
final


# In[14]:


X = final.drop('price', axis='columns')
X


# In[15]:


y = final.price


# In[18]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[19]:


model.fit(X,y)


# In[20]:


model.predict(X) # 2600 sqr ft home in new jersey


# In[21]:


model.score(X,y)


# In[22]:


model.predict([[3400,0,0]]) # 3400 sqr ft home in west windsor


# In[23]:


model.predict([[2800,0,1]]) # 2800 sqr ft home in robbinsville


# <h2 style='color:purple'>Using sklearn OneHotEncoder</h2>

# First step is to use label encoder to convert town names into numbers

# In[24]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[25]:


dfle = df
dfle.town = le.fit_transform(dfle.town)
dfle


# In[26]:


X = dfle[['town','area']].values


# In[27]:


X


# In[28]:


y = dfle.price.values
y


# Now use one hot encoder to create dummy variables for each of the town

# In[29]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('town', OneHotEncoder(), [0])], remainder = 'passthrough')


# In[30]:


X = ct.fit_transform(X)
X


# In[31]:


X = X[:,1:]


# In[32]:


X


# In[33]:


model.fit(X,y)


# In[34]:


model.predict([[0,1,3400]]) # 3400 sqr ft home in west windsor


# In[35]:


model.predict([[1,0,2800]]) # 2800 sqr ft home in robbinsville


# 

# 
