#!/usr/bin/env python
# coding: utf-8

# # Zillow

# ### Wrangling the Zillow data
# 
# #### Aquuires and prepares Zillow data

# ## Imports

# In[1]:


import numpy as np
import pandas as pd

# Visualizing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

# default pandas decimal number display format
pd.options.display.float_format = '{:.2f}'.format


# Stats
import scipy.stats as stats

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

#My Files
import env


# ## Acquire

# In[2]:


def get_db_url(database):
    from env import host, user, password
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    return url

# acquire zillow data using the query
def get_zillow():
    # SQL Query
    sql = '''

    SELECT prop.*, 
           pred.logerror, 
           pred.transactiondate, 
           air.airconditioningdesc, 
           arch.architecturalstyledesc, 
           build.buildingclassdesc, 
           heat.heatingorsystemdesc, 
           landuse.propertylandusedesc, 
           story.storydesc, 
           construct.typeconstructiondesc 
    FROM   properties_2017 prop  
           INNER JOIN (SELECT parcelid,
                       Max(transactiondate) transactiondate 
                       FROM   predictions_2017 
  
                       GROUP  BY parcelid) pred 
                   USING (parcelid)
                   
                            JOIN predictions_2017 as pred USING (parcelid, transactiondate)
           LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
           LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
           LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
           LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
           LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
           LEFT JOIN storytype story USING (storytypeid) 
           LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
    WHERE  prop.latitude IS NOT NULL 
           AND prop.longitude IS NOT NULL
           AND pred.id IN (SELECT MAX(id)
           FROM predictions_2017
           GROUP BY parcelid
           HAVING MAX(transactiondate));
       
       
'''
    url = get_db_url('zillow')
    zillow_df = pd.read_sql(sql, url, index_col='id')
    return zillow_df


# In[3]:


df = get_zillow()


# In[4]:


df.head()


# ## Prepare

# In[15]:


# a function to drop missing values based on a threshold
def handle_missing_values(df, prop_required_row = 0.5, prop_required_col = 0.5):
    ''' function which takes in a dataframe, required notnull proportions of non-null rows and columns.
    drop the columns and rows columns based on theshold:'''
    
    #drop columns with nulls
    threshold = int(prop_required_col * len(df.index)) # Require that many non-NA values.
    df.dropna(axis = 1, thresh = threshold, inplace = True)
    
    #drop rows with nulls
    threshold = int(prop_required_row * len(df.columns)) # Require that many non-NA values.
    df.dropna(axis = 0, thresh = threshold, inplace = True)
    
    
    return df


# In[13]:


def wrangle_zillow():
    # read saved .csv
    df = pd.read_csv('zillow.csv')
    
    # propertylandusetypeid that can be considered "single unit" to df
    single_unit = [261, 262, 263, 264, 268, 273, 275, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_unit)]
    
    # df with bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet > 0
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & (df.calculatedfinishedsquarefeet>0)]

    
    # drop missing values based on a threshold
    df = handle_missing_values(df)
   
    # drop unnecessary columns
    df = df.drop(columns=['calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 'heatingorsystemtypeid', 'propertyzoningdesc', 'censustractandblock','propertycountylandusecode', 'propertylandusetypeid', 'propertylandusedesc', 'unitcnt','heatingorsystemdesc'])
    
    # drop null rows for specific columns only
    df = df[df.regionidzip.notnull()]
    df = df[df.yearbuilt.notnull()]
    df = df[df.structuretaxvaluedollarcnt.notnull()]
    df = df[df.taxvaluedollarcnt.notnull()]
    df = df[df.landtaxvaluedollarcnt.notnull()]
    df = df[df.taxamount.notnull()]

    # fill NaNs with mode
    df.buildingqualitytypeid.mode()[0]
    df['buildingqualitytypeid'] = df.buildingqualitytypeid.fillna(df.buildingqualitytypeid.mode()[0])
    df.lotsizesquarefeet.mode()[0]
    df['lotsizesquarefeet'] = df.lotsizesquarefeet.fillna(df.lotsizesquarefeet.mode()[0])
    df.regionidcity.mode()[0]
    df['regionidcity'] = df.regionidcity.fillna(df.regionidcity.mode()[0])

    
    # crete column called county that lists column type based on fips
    df['county'] = df['fips'].apply(
    lambda x: 'Los Angeles' if x == 6037\
    else 'Orange' if x == 6059\
    else 'Ventura')
    
    # check for outliers
    df = df[df.taxvaluedollarcnt < 5_000_000]
    df[df.calculatedfinishedsquarefeet < 8000]
    
    
    # drop nulls to make sure none were missed
    df = df.dropna()
    
    return df
    


# In[14]:


df = wrangle_zillow()
df.head()


# In[ ]:




