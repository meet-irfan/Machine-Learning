#!/usr/bin/env python
# coding: utf-8

# In[24]:


from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# In[25]:


iris = load_iris()


# In[26]:


#Use Logistic Regression 
l_scores = cross_val_score(LogisticRegression(), iris.data, iris.target)
l_scores


# In[27]:


np.average(l_scores)


# In[28]:


#DecisionTreeClassifier
d_scores = cross_val_score(DecisionTreeClassifier(), iris.data, iris.target)
d_scores


# In[29]:


np.average(d_scores)


# In[30]:


#SCV
s_scores = cross_val_score(SVC(), iris.data, iris.target)
s_scores


# In[31]:


np.average(s_scores)


# In[32]:


#RandomForestClassifier
r_scores = cross_val_score(RandomForestClassifier(), iris.data, iris.target)
r_scores


# In[33]:


np.average(r_scores)


# In[36]:


#Best score so far is from Logistic Regression: 0.9733333333333334


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




