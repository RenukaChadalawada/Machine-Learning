#!/usr/bin/env python
# coding: utf-8

# # Author : Renuka Chadalawada

# # Task2: Prediction using Unsupervised ML
# 
# From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.
#     

# ### Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Load the dataset & read

# In[2]:


dataset=pd.read_csv("D:\Data sciece SP project\Iris.csv")


# In[3]:


dataset


# ### Check the if null values

# In[4]:


dataset.isnull().any()


# ### Splitting the Independent Variables

# In[5]:


x=dataset.iloc[:,1:5]


# In[6]:


x


# ### Elbow Method

# In[7]:


k=np.arange(1,11)


# In[8]:


k


# In[9]:


wcss=[]


# In[10]:


from sklearn.cluster import KMeans


# In[11]:


#n_cluster -- No of clusters
for k_value in k:
    # Creating the instance of KMeans
    k_means_model=KMeans(n_clusters=k_value)
    # Training the KMeans Model
    k_means_model.fit(x)
    #Find WCSS value
    #Inertia -- WCS value
    wcss.append(k_means_model.inertia_)


# In[12]:


wcss


# In[13]:


plt.plot(k,wcss,marker="o")
plt.xlabel("No.of clusters")
plt.xticks(k)
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()


# In[14]:


# Optimal no.of clusters = 3


# ### Building the model for K Means

# In[15]:


k_means_model=KMeans(n_clusters=3,random_state=0)


# In[16]:


k_means_model.fit(x)


# In[17]:


y_pred=k_means_model.predict(x)


# In[18]:


y_pred


# In[19]:


# Of zeroth cluster
x[y_pred==0]


# ### Visualisation using first two Columns

# In[20]:


plt.scatter(x[y_pred==0]["SepalLengthCm"],x[y_pred==0]["SepalWidthCm"],label="Iris-setosa")
plt.scatter(x[y_pred==1]["SepalLengthCm"],x[y_pred==1]["SepalWidthCm"],label="Iris-versicolour")
plt.scatter(x[y_pred==2]["SepalLengthCm"],x[y_pred==2]["SepalWidthCm"],label="Iris-virginica")
plt.scatter(k_means_model.cluster_centers_[:,0],k_means_model.cluster_centers_[:,1],color="black",s=100)
plt.title("K-means Clustering")
plt.xlabel("SepalLengthCm")
plt.ylabel("SepalWidthCm")
plt.legend()
plt.show()


# ### Real Time Predictions

# In[21]:


k_means_model.predict([[3,1.2,2.8,4.0]])


# # Conclustion

# From the given ‘Iris’ dataset, predict the optimum number of clusters are 3 and represented it visually with first two columns -  Sepalwidth and Sepallength.
