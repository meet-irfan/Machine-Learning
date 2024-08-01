#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


dataset=pd.read_csv("Breast Cancer.csv")


# In[4]:


dataset.head()


# In[6]:


# loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)
data_frame.head()


# In[7]:


# adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target


# In[8]:


# print last 5 rows of the dataframe
data_frame.tail()


# In[9]:


data_frame.shape


# In[10]:


data_frame.info()


# In[12]:


data_frame.isnull().sum()


# In[13]:


data_frame.describe()


# In[14]:


# checking the distribution of Target Varibale
data_frame['label'].value_counts()


# 1 --> Benign
# 
# 0 --> Malignant

# In[15]:


data_frame.groupby('label').mean()


# Separating the features and target

# In[16]:


X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']


# In[17]:


print(X)


# In[19]:


print(Y)


# Splitting the data into training data & Testing data

# In[20]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[21]:


print(X.shape, X_train.shape, X_test.shape)


# In[22]:


model = LogisticRegression()
model.fit(X_train, Y_train)


# Model Evaluation
# 

# In[23]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)


# In[25]:


print('Accuracy on Training Data: = ', training_data_accuracy)


# In[26]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)


# In[27]:


print('Accuracy on Test Data: = ', test_data_accuracy)


# # Building a Predictive System

# In[28]:


input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast cancer is Malignant')

else:
  print('The Breast Cancer is Benign')



# In[ ]:




