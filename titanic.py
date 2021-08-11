#!/usr/bin/env python
# coding: utf-8
# Description: This program predicts if a passenger will survive on the titanic
# In[73]:


#Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#Load the data
titanic = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset1/master/titanic_train.csv')


# In[74]:


titanic.head(10)


# In[75]:


titanic.describe()


# In[76]:


#Get a count of the number of survivors  
titanic['Survived'].value_counts()


# In[77]:


titanic['Survived'].value_counts().keys()

Visualize the number of survivors on board the Titanic in this data set.
# In[38]:


#Visualize the count of number of survivors
sns.countplot(titanic['Survived'],label="Count")


# In[78]:


titanic['Pclass'].value_counts()


# In[79]:


sns.countplot(titanic['Pclass'],label="Count")


# In[80]:


titanic['Sex'].value_counts()


# In[42]:


sns.countplot(titanic['Sex'],label="Count")


# In[81]:


titanic['Age'].value_counts()


# In[82]:


sns.countplot(titanic['Age'],label="Count")


# In[83]:


sum(titanic['Survived'].isnull())


# In[84]:


sum(titanic['Age'].isnull())


# In[85]:


titanic = titanic.dropna()


# In[96]:


sum(titanic['Age'].isnull())


# In[97]:


X_train = titanic [['Age']]
y_train = titanic [['Survived']]


# In[98]:


from sklearn.tree import DecisionTreeClassifier


# In[102]:


dtc = DecisionTreeClassifier()


# In[101]:


dtc.fit(X_train,y_train)

