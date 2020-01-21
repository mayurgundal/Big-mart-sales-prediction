# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:19:26 2019

@author: Mayur
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("train_kOBLwZA.csv")

for x in df.columns:
    print(f"{x} - {df[x].nunique()}")
    
des = df.describe()

#All the categorical feature unique elements are matching with
#both train and test sets


cat_col=[]
num_col=[]
for x in df.columns:
    if type(df[x][0]) == str:
        cat_col.append(x)
    elif type(df[x][0]) == int:
        num_col.append(x)
    else:
        num_col.append(x)
cat_col.pop(0)
        
###########################
""" Exploratory Data Analaysis"""
plt.figure(figsize=(40,40))
plt.subplot(3,1,1)
sns.barplot(x="Item_Fat_Content",y="Item_Outlet_Sales",data=df)
plt.title("Fat")
plt.subplot(3,1,2)
sns.barplot(x="Item_Type",y="Item_Outlet_Sales",data=df)
plt.title("Item type")
plt.subplot(3,1,3)
sns.barplot(x="Outlet_Identifier",y="Item_Outlet_Sales",data=df)
plt.title("Outlet")
plt.tight_layout()

plt.figure()
sns.barplot(x="Outlet_Size",y="Item_Outlet_Sales",data=df)

plt.figure()
sns.barplot(x="Outlet_Location_Type",y="Item_Outlet_Sales",data=df)

plt.figure()
sns.barplot(x="Outlet_Type",y="Item_Outlet_Sales",data=df)

na = df.isnull().sum()
sns.barplot(x="Outlet_Type",hue="Outlet_Size",y="Item_Outlet_Sales",data=df)

plt.figure()
sns.distplot(df.Item_Weight)
"""Normally distributed"""

plt.figure()
sns.distplot(df.Item_Visibility)
"""Right skewed, small values"""

plt.figure()
sns.distplot(df.Item_MRP)
"""Peaks and lows, not normally distributed"""

plt.figure()
sns.distplot(df.Item_MRP)

plt.figure()
sns.barplot(x="Outlet_Establishment_Year", y ="Item_Outlet_Sales",data=df)





#===============================================================================
# Filling missing value
df["Outlet_Size"] = df["Outlet_Size"].fillna(df["Outlet_Size"].mode()[0])
df["Item_Weight"] = df["Item_Weight"].fillna(df["Item_Weight"].mean())

df.isnull().values.any()

#####Encoding###############
dataset=df["Item_Identifier"]
for x in cat_col:
    x = pd.get_dummies(df[x],drop_first=True)
    dataset = pd.concat([dataset,x],axis=1)
for x in num_col:
    dataset = pd.concat([dataset,df[x]],axis=1)
    
dummy = pd.get_dummies(df["Outlet_Establishment_Year"],drop_first=True)
dataset = pd.concat([dataset,dummy],axis=1)
dataset.drop(["Outlet_Establishment_Year"],axis=1,inplace=True)

X = dataset.drop(["Item_Identifier","Item_Outlet_Sales"],axis=1)
y = dataset["Item_Outlet_Sales"]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
cols_to_scale = ['Item_Weight','Item_Visibility','Item_MRP']
X[cols_to_scale] = sc.fit_transform(X[cols_to_scale])



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)




from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=700,max_depth = 5,random_state=0)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(y_test,y_pred_xgb)))

from sklearn.model_selection import RandomizedSearchCV
parameters = {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
              "n_estimators" : np.arange(100,1600,100),
 "max_depth": [ 2,3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [1,2, 3,4, 5, 7],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }
rdmsearch = RandomizedSearchCV(xgb,
                           param_distributions = parameters,
                           scoring = 'neg_mean_absolute_error',
                           cv = 10,
                           n_jobs = 4,random_state=42)
rdmsearch = rdmsearch.fit(X,y)
best_accuracy = rdmsearch.best_score_
best_parameters = rdmsearch.best_params_
print(best_accuracy)
rdmsearch.best_estimator_

xgb = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.4, gamma=0.4,
       importance_type='gain', learning_rate=0.05, max_delta_step=0,
       max_depth=6, min_child_weight=3, missing=None, n_estimators=200,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,y_pred_xgb)))

test = pd.read_csv("test.csv")
test["Outlet_Size"] = test["Outlet_Size"].fillna(test["Outlet_Size"].mode()[0])
test["Item_Weight"] = test["Item_Weight"].fillna(test["Item_Weight"].mean())
cat_col=[]
num_col=[]
for x in test.columns:
    if type(test[x][0]) == str:
        cat_col.append(x)
    elif type(test[x][0]) == int:
        num_col.append(x)
    else:
        num_col.append(x)

d_test=test["Item_Identifier"]
cat_col.pop(0)
for x in cat_col:
    x = pd.get_dummies(test[x],drop_first=True)
    d_test = pd.concat([d_test,x],axis=1)
for x in num_col:
    d_test = pd.concat([d_test,test[x]],axis=1)
    
dum = pd.get_dummies(test["Outlet_Establishment_Year"],drop_first=True)
d_test = pd.concat([d_test,dum],axis=1)
d_test.drop(["Outlet_Establishment_Year"],axis=1,inplace=True)

cols_to_scale = ['Item_Weight',
 'Item_Visibility',
 'Item_MRP']
X[cols_to_scale] = sc.fit(X[cols_to_scale])
d_test[cols_to_scale] = sc.fit_transform(d_test[cols_to_scale])


Xt = d_test.drop(["Item_Identifier"],axis=1)

y_pred = xgb.predict(Xt)
test = pd.read_csv("test.csv")
pred = pd.DataFrame(y_pred)
ide = test[["Item_Identifier","Outlet_Identifier"]]
sub = pd.concat([ide,pred],axis=1)
sub.columns = ["Item_Identifier","Outlet_Identifier","Item_Outlet_Sales"]
sub.to_csv("sample_submission.csv",index=False)