#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Se ignoran los "FutureWarnings" molestos
import warnings
from sklearn.linear_model import LinearRegression
warnings.simplefilter(action='ignore', category=FutureWarning) 


# In[16]:


# se calcula porcentaje de valores null por columna y se dropean las columnas con porcentaje de nan superior a 10%
#aca hay que verificar que cada columna dropeada no contiene valores excluyentes con algun otro dato -->estos pueden ser recuperados
#hace falta agregar un modulo que hace siguiente calculo: si Cloud9am y Cloud3pm son distintos de nan y Sunshine es nan, se reemplaza o se complementa
#se puede quitar las ultimas dos columnas donde se expresa literalmente si llovió, alla tenemos la columna de rainfall con milimitros.
# # pressure y evaporation no parecen tener sustitutos
def drop_nans(ds):
   per=[]
   for column in ds:
      tot=ds[column].isnull().sum()
      per=tot/len(ds[column])*100
      if per > 10:
         print(column,"porcentaje de Null",per,"dropea")
         ds = ds.drop(column, 1)
      else:
         print(column,"cantidad de Null",tot,"porcentaje de Null",per, "se conserva")
   return ds


# ##en caso de conservar las ultimas dos columnas, se puede reemplazar los nan usando knn 
# #sobre columna de rainfall y posteriormente usar el siguiente comando
# #dataset.replace({"Yes": 1, "No": 0})
# #print("columna RainToday contiene: \n",pd.value_counts(dataset.RainToday),"columna RainTomorrow contiene: \n",pd.value_counts(dataset.RainTomorrow))

# In[21]:


# Codificación de la categoría DATE en dos columnas: Year y MonthDay que es un número con decimales entre 1 y 13

def date2columns(ds):
    if 'Date' in ds.columns:
        date = ds['Date'].str.split(pat='-', expand = True)
        date.shape
        year = np.uint16(date[0])
        month_day = np.float32(date[1]) + np.divide(np.float32(date[2]),32)
        ds = ds.drop('Date', 1)
        ds.insert(0, 'Year', year)
        ds.insert(1, 'MonthDay', month_day)
    return ds


# In[19]:


# Codificación de la categoría 'Location'
def geoloc(ds):
    import geolocators
    import importlib
    importlib.reload(geolocators)

    # Se agregan las columnas nuevas
    if 'Latitude' in ds.columns:
        print ('la columna latitude ya existe')
    else:
        ds.insert(2, 'Latitude', np.empty(len(ds['Location'])))
    if 'Longitude' in ds.columns:
        print ('la columna longitude ya existe')
    else:
        ds.insert(3, 'Longitude', np.empty(len(ds['Location'])))

    print('El dataset tiene {} localidades distintas'.format(ds['Location'].unique().shape[0]))
    for loc in ds['Location'].unique():
        print(geolocators.coordinates(loc), loc)
        ds.Latitude.replace()
        ds['Latitude'] = ds['Latitude'].where(ds['Location'] != loc, geolocators.coordinates(loc)[0])
        ds['Longitude'] = ds['Longitude'].where(ds['Location'] != loc, geolocators.coordinates(loc)[1])
    if 'Location' in ds.columns: ds = ds.drop('Location', 1)
    return ds 



# In[20]:


#definimos encoder y normalizamos los datos (menos la fecha), generamos otra tabla:
def normal(ds):
   import sklearn.preprocessing as preprocessing
   from sklearn.preprocessing import StandardScaler
   ss = StandardScaler()
   enc = preprocessing.OrdinalEncoder()
   dsscal_=ds.copy()
   k=0

   for column in ds:
      if np.dtype(np.dtype(ds[column])) == "object" and column not in ["Year","MonthDay"]:
         dsscal_.insert(k, f'{column}_enc_scal', ss.fit_transform(enc.fit_transform(ds[column].values.reshape(-1, 1))))
         dsscal_ = dsscal_.drop(column, 1)

      elif column not in ["Year","MonthDay"]:
         dsscal_.insert(k, f'{column}_scal', ss.fit_transform(ds[column].values.reshape(-1, 1)))
         dsscal_ = dsscal_.drop(column, 1)

      k=k+1
   return ds


# Multiple Imputation by Chained Equations (MICE)

def impute_column(df, col_to_predict, feature_columns):
    df3 = df.copy()
    for column in df3[feature_columns].columns:
        df3.loc[df3[column].isna(), column] = df3[column].mean()

        
    nan_rows = np.where(np.isnan(df3[col_to_predict]))
    all_rows = np.arange(0,len(df3))
    train_rows_idx = np.argwhere(~np.isin(all_rows,nan_rows)).ravel()
    pred_rows_idx =  np.argwhere(np.isin(all_rows,nan_rows)).ravel()
    print(len(train_rows_idx))
    print(len(pred_rows_idx))
    X_train,y_train = df3[feature_columns].iloc[train_rows_idx],df3[col_to_predict].iloc[train_rows_idx]
    X_pred = df3[feature_columns].iloc[pred_rows_idx]

    model = LinearRegression()
    model.fit(X_train,y_train)
    #print(X_pred.values)
    df3[col_to_predict].iloc[pred_rows_idx] = model.predict(X_pred.values)#.reshape(1,-1))
    return df3
       
def impute_column_int(df, col_to_predict, feature_columns):
    df3 = df.copy()
    for column in df3[feature_columns].columns:
        df3.loc[df3[column].isna(), column] = df3[column].mean()

        
    nan_rows = np.where(np.isnan(df3[col_to_predict]))
    all_rows = np.arange(0,len(df3))
    train_rows_idx = np.argwhere(~np.isin(all_rows,nan_rows)).ravel()
    pred_rows_idx =  np.argwhere(np.isin(all_rows,nan_rows)).ravel()
    print(len(train_rows_idx))
    print(len(pred_rows_idx))
    X_train,y_train = df3[feature_columns].iloc[train_rows_idx],df3[col_to_predict].iloc[train_rows_idx]
    X_pred = df3[feature_columns].iloc[pred_rows_idx]

    model = LinearRegression()
    model.fit(X_train,y_train)
    #print(X_pred.values)
    df3[col_to_predict].iloc[pred_rows_idx] = np.int8(model.predict(X_pred.values))#.reshape(1,-1))
    return df3

# In[ ]:




