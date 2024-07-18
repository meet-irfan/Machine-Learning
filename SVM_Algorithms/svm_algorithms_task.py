#!/usr/bin/env python
# coding: utf-8

# <h2 style='color:blue' align="center">Support Vector Machine Tutorial Using Python Sklearn</h2>

# In[5]:


import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()


# <img height=300 width=300 src="iris_petal_sepal.png" />

# In[2]:


iris.feature_names


# In[3]:


iris.target_names


# In[6]:


df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()


# In[8]:


df['target'] = iris.target
df.head()


# In[9]:


df[df.target==1].head()


# In[10]:


df[df.target==2].head()


# In[11]:


df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])
df.head()


# In[13]:


df[45:55]


# In[15]:


df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]


# In[14]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# **Sepal length vs Sepal Width (Setosa vs Versicolor)**

# In[17]:


plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')


# **Petal length vs Pepal Width (Setosa vs Versicolor)**

# In[18]:


plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')


# **Train Using Support Vector Machine (SVM)**

# In[49]:


from sklearn.model_selection import train_test_split


# In[50]:


X = df.drop(['target','flower_name'], axis='columns')
y = df.target


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[52]:


len(X_train)


# In[53]:


len(X_test)


# In[75]:


from sklearn.svm import SVC
model = SVC()


# In[76]:


model.fit(X_train, y_train)


# In[77]:


model.score(X_test, y_test)


# In[78]:


model.predict([[4.8,3.0,1.5,0.3]])


# **Tune parameters**

# **1. Regularization (C)**

# In[97]:


model_C = SVC(C=1)
model_C.fit(X_train, y_train)
model_C.score(X_test, y_test)


# In[106]:


model_C = SVC(C=10)
model_C.fit(X_train, y_train)
model_C.score(X_test, y_test)


# **2. Gamma**

# In[103]:


model_g = SVC(gamma=10)
model_g.fit(X_train, y_train)
model_g.score(X_test, y_test)


# **3. Kernel**

# In[104]:


model_linear_kernal = SVC(kernel='linear')
model_linear_kernal.fit(X_train, y_train)


# In[105]:


model_linear_kernal.score(X_test, y_test)


# 

# 
