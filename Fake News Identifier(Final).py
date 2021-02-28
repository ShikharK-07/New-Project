#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Necessary Imports
import numpy as np
import pandas as pd
import itertools
import re
import string


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report


# In[3]:


pwd


# In[4]:


df_fake=pd.read_csv('C:\\Users\\OK\\Documents\\Fake News Dataset\\MyProject2\Fake.csv')
df_true=pd.read_csv('C:\\Users\\OK\\Documents\\Fake News Dataset\\MyProject2\True.csv')
df_fake.head(10)


# In[5]:


df_true.head(10)


# In[6]:


df_fake["class"]=0
df_true["class"]=1


# In[7]:


df_fake.shape,df_true.shape


# In[8]:


df_fake_manual_testing=df_fake.tail(10)
df_fake_manual_testing


# In[9]:


df_true_manual_testing=df_true.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis=0, inplace=True) 


# In[10]:


for i in range(21416,21406,-1):
    df_true.drop([i], axis=0, inplace=True)


# In[11]:


df_manual_testing=pd.concat([df_fake_manual_testing,df_true_manual_testing], axis=0)
df_manual_testing.to_csv("C:\\Users\\OK\\Documents\\Fake News Dataset\\MyProject2\manual_testing.csv")


# In[12]:


df_merge=pd.concat([df_fake,df_true],axis=0)
df_merge.head(10)


# In[13]:


#dropping columns ["title","subject","date"]
df=df_merge.drop( ["title","subject","date"],axis=1)
df.head(10)


# In[14]:


#sampling
df=df.sample(frac=1)
df.head(10)


# In[15]:


#Finding missing values
df.isnull().sum()


# In[16]:


#removing special/unnecessary texts
def word_drop(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub("\\W"," ",text)
    text=re.sub('https:?://\S+|www\.\S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('{%s]' % re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    return text


# In[17]:


df["text"]=df["text"].apply(word_drop)
df.head(10)


# In[18]:


x = df["text"]
y = df["class"]


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25)


# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorization=TfidfVectorizer()
xv_train=vectorization.fit_transform(x_train)
xv_test=vectorization.transform(x_test)


# In[21]:


from sklearn.linear_model import LogisticRegression


# In[22]:


LR=LogisticRegression()
LR.fit(xv_train,y_train)


# In[23]:


LR.score(xv_test,y_test)


# In[24]:


pred_LR=LR.predict(xv_test)


# In[25]:


print(classification_report(y_test,pred_LR))


# In[26]:


from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()


# In[27]:


DT.fit(xv_train,y_train)


# In[28]:


DT.score(xv_test,y_test)


# In[29]:


pred_DT=DT.predict(xv_test)
print(classification_report(y_test,pred_DT))


# In[30]:


####Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
GBC=GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train,y_train)


# In[31]:


GBC.score(xv_test,y_test)


# In[32]:


pred_GBC=GBC.predict(xv_test)


# In[33]:


print(classification_report(y_test,pred_GBC))


# In[34]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier


# In[35]:


RFC=RandomForestClassifier(random_state=0)
RFC.fit(xv_train,y_train)


# In[36]:


pred_RFC=RFC.predict(xv_test)
print(classification_report(y_test,pred_RFC))


# In[37]:


#Manual Testing
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(word_drop) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]), 
                                                                                                              output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_GBC[0]), 
                                                                                                              output_lable(pred_RFC[0])))


# In[48]:



news = str(input())
manual_testing(news)


# In[ ]:




