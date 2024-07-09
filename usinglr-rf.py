#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sampleSubmission.csv')


# In[4]:


# 1. Remove outliers
train = train[train['weather'] != 4]


# In[5]:


# 2. concat data between train and test 
data = pd.concat([train, test], ignore_index=True)
data.tail()


# In[6]:


# 3. add features 
data['datetime'] = pd.to_datetime(data['datetime'])
data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['hour'] = data['datetime'].dt.hour
data['weekday'] = data['datetime'].dt.weekday


# In[7]:


# 4. remove unrelated features 
data.drop(['casual', 'registered', 'datetime', 'month', 'windspeed'], axis=1, inplace=True)
data.head()


# In[8]:


#5. split datasets
train = data[~pd.isnull(data['count'])]
test = data[pd.isnull(data['count'])]

X_train = train.drop(['count'], axis=1)
X_test = test.drop(['count'], axis=1)

y_train = train['count']


# In[9]:


import numpy as np 

def rmsle(y_true, y_pred, convertExp=True):
    if convertExp:
        y_true = np.exp(y_true)
        y_pred = np.exp(y_pred)
    
    log_true = np.nan_to_num(np.log(y_true + 1))
    log_pred = np.nan_to_num(np.log(y_pred + 1)) # convert missing value to 0
    
    output = np.sqrt(np.mean((log_true - log_pred) ** 2))
    return output


# In[10]:


from sklearn.linear_model import LinearRegression 

linear_reg_model = LinearRegression()


# In[11]:


log_y = np.log(y_train)
linear_reg_model.fit(X_train, log_y)


# In[12]:


preds = linear_reg_model.predict(X_train)


# In[13]:


print(f'RMSLE: {rmsle(log_y, preds, True):.4f}')


# In[14]:


test_preds = linear_reg_model.predict(X_test)

submission['count'] = np.exp(test_preds)
submission.to_csv('submission.csv', index=False)


# In[15]:


import numpy as np 
from sklearn import metrics 
def rmsle1(y_true, y_pred, convertExp=True):
    if convertExp:
        y_true = np.exp(y_true)
        y_pred = np.exp(y_pred)
    
    log_true = np.nan_to_num(np.log(y_true + 1))
    log_pred = np.nan_to_num(np.log(y_pred + 1)) # convert missing value to 0
    
    output = np.sqrt(np.mean((log_true - log_pred) ** 2))
    return output


# In[16]:


rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)


# In[17]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
randomforest = RandomForestRegressor()

rf_params = {'random_state': [42], 'n_estimators': [10, 20, 140]}
gridsearch_random_forest = GridSearchCV(estimator=randomforest, 
                                        param_grid=rf_params, 
                                        scoring=rmsle_scorer, 
                                        cv=5)

log_y = np.log(y_train)
gridsearch_random_forest.fit(X_train, log_y)
print(f'Best Parameter: {gridsearch_random_forest.best_params_}')


# In[18]:


train_preds = gridsearch_random_forest.best_estimator_.predict(X_train)

print(f'RMSLE of random forest: {rmsle(log_y, train_preds, True):.4f}')


# In[19]:


import seaborn as sns 
import matplotlib.pyplot as plt 

test_preds1 = gridsearch_random_forest.best_estimator_.predict(X_test)


figure, axes = plt.subplots(ncols=2)
figure.set_size_inches(10, 4)

sns.histplot(y_train, bins=50, ax=axes[0])
axes[0].set_title('Train Data Distribution')

sns.histplot(np.exp(test_preds), bins=50, ax = axes[1])
axes[1].set_title('Test Data Distribution')


# In[21]:


submission['count1'] = np.exp(test_preds1)
submission.to_csv('submission.csv', index=False)


# In[ ]:




