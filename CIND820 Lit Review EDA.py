#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


cust_sat_df = pd.read_csv('Airline_customer_satisfaction.csv')
# head of df
print(cust_sat_df.head())


# In[3]:


# summary of df
print(cust_sat_df.info())


# In[4]:


# Summary stats
print(cust_sat_df.describe())


# In[17]:


# missing vals
print(cust_sat_df.isnull().sum())


# In[5]:


# Remove null rows in "arrival delay in mins"
cust_sat_df_cleaned = cust_sat_df.dropna(subset=['Arrival Delay in Minutes'])

print(cust_sat_df_cleaned)


# In[20]:


# Check for missing values
print(cust_sat_df_cleaned.isnull().sum())


# In[6]:


# Check for duplicate rows
print(cust_sat_df_cleaned.duplicated().sum())


# In[22]:


cust_sat_df_cleaned.info()


# In[7]:


# Histogram
cust_sat_df_cleaned.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()


# In[23]:


print(cust_sat_df_cleaned.dtypes)


# In[15]:


numerical_vars = cust_sat_df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()

for var in numerical_vars:
    plt.figure(figsize=(6, 4))
    plt.hist(cust_sat_df_cleaned[var], bins=30, edgecolor='k')
    plt.title(f'Histogram of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.show()


# In[14]:


categorical_vars = cust_sat_df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()

# Plot pie chart and bar graph for each categorical variable
for var in categorical_vars:
    # Pie chart
    plt.figure(figsize=(4, 4))
    cust_sat_df_cleaned[var].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title(f'Distribution of {var}')
    plt.ylabel('')
    plt.show()

    # Bar graph
    plt.figure(figsize=(6, 4))
    cust_sat_df_cleaned[var].value_counts().plot(kind='bar', edgecolor='k')
    plt.title(f'Bar Graph of {var}')
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.show()


# In[32]:


for var in numerical_vars:
    plt.figure(figsize=(4, 2))
    plt.boxplot(cust_sat_df_cleaned[var])
    plt.title(f'Outlier Graph for {var}')
    plt.ylabel(var)
    plt.show()


# In[35]:


import seaborn as sns
numerical_df = cust_sat_df_cleaned.select_dtypes(include=['int64', 'float64'])

correlation_matrix = numerical_df.corr()

import seaborn as sns
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()


# In[ ]:




