#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Sample data
data = np.array([[2.5, 2.4],
                 [0.5, 0.7],
                 [2.2, 2.9],
                 [1.9, 2.2],
                 [3.1, 3.0],
                 [2.3, 2.7],
                 [2, 1.6],
                 [1, 1.1],
                 [1.5, 1.6],
                 [1.1, 0.9]])

# Standardize the data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

print("Standardized Data:\n", data_standardized)


# # Compute the Covariance Matrix

# In[2]:


# Compute the covariance matrix
cov_matrix = np.cov(data_standardized, rowvar=False)

print("Covariance Matrix:\n", cov_matrix)


# # Compute the Eigenvalues and Eigenvectors

# In[3]:


# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)


# # Sort Eigenvalues and Eigenvectors

# In[4]:


# Sort the eigenvalues and eigenvectors
sorted_index = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_index]
sorted_eigenvectors = eigenvectors[:, sorted_index]

print("Sorted Eigenvalues:\n", sorted_eigenvalues)
print("Sorted Eigenvectors:\n", sorted_eigenvectors)


# In[7]:


# Select the Top k Eigenvectors
# Select the top k eigenvectors (let's use k=1 for this example)
k = 2
eigenvector_subset = sorted_eigenvectors[:, :k]

print("Top k Eigenvectors:\n", eigenvector_subset)


# In[8]:


# Transform the data
data_transformed = np.dot(data_standardized, eigenvector_subset)

print("Transformed Data:\n", data_transformed)


# In[ ]:




