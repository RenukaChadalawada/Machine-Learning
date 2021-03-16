#!/usr/bin/env python
# coding: utf-8

# # Author : Renuka Chadalawada

# # Task 6: Prediction using Decision Tree Algorithm
# 
#    

# ### Create the Decision Tree classifier and visualize it graphically.

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dataset=pd.read_csv(r"D:\Data sciece SP project\Iris.csv")


# In[3]:


dataset=dataset.drop("Id",axis=1)


# In[4]:


dataset


# ### Check the if null values

# In[5]:


dataset.isnull().any()


# ### Split Dependent and Independent variables

# In[6]:


x=dataset.iloc[:,0:4] # Independent Variable


# In[7]:


x.head()


# In[8]:


y=dataset.iloc[:,4] # Dependent Variable


# In[9]:



y.head()


# 
# 
# 
# 
# ### Split data into train and test

# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=26)


# In[12]:


x_train.shape


# In[13]:


x_test.shape


# ### Build the Model

# In[14]:


from sklearn.tree import DecisionTreeClassifier


# In[15]:


dt_model=DecisionTreeClassifier(criterion="entropy")


# In[16]:


dt_model.fit(x_train,y_train)


# ### Prediction

# In[17]:


#Testing Phase
y_pred=dt_model.predict(x_test)


# In[18]:


y_pred


# ### Evaluation

# In[19]:


from sklearn.metrics import accuracy_score


# In[20]:


print("Test Accuracy Score::",accuracy_score(y_test,y_pred)*100,"%")


# In[21]:


y_pred_train=dt_model.predict(x_train)


# In[22]:


print("Train Accuracy Score::",accuracy_score(y_train,y_pred_train)*100,"%")


# In[23]:


from mlxtend.plotting import plot_decision_regions


# ### plot Decission Tree

# In[24]:


from sklearn.tree import plot_tree


# In[25]:


plt.figure(figsize=(10,15))
plot_tree(dt_model,fontsize=10,feature_names=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"],filled=True)
plt.show()


# ### Real Time Predictions

# In[26]:


X=[[6.4,1.8 ,6.6 ,2.1]]


# In[27]:


prediction=dt_model.predict(X)


# In[28]:


prediction


# In[29]:


X=[[3.4,1.8 ,2.6 ,3.1]]


# In[30]:


prediction=dt_model.predict(X)


# In[31]:


prediction


# # Conclusion

# ##### According to problem statement if we feed any new data to this classifier, it is able to predict the right class accordingly.
