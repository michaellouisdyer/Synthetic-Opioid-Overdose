
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.plotly as py
import plotly.offline as plto
import plotly.figure_factory as ff
import statsmodels.api as sm
from prep import get_data
from fancyimpute import KNN

from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from EDA import impute_df, drop_nulls
from statsmodels.stats.outliers_influence import variance_inflation_factor as c_vif

mpl.rcParams.update({
    'font.size'           : 20.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'medium',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'medium',
})

def scale_df(df,scaler):
    scaler.fit(df)
    return pd.DataFrame(data=scaler.transform(df), columns=df.columns, index=df.index)

def vif(X):
    vif = pd.DataFrame()
    vif["VIF Factor"] = [c_vif(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns
    return vif
#main

mcd, mcd_main, mcd_wide, mcd_2015, mcd_2016 =  get_data()
T40 = drop_nulls(mcd_main,['T40.4'])
T40_complete = impute_df(T40, KNN(5))


scaler = StandardScaler()

#check for heteroscedasticity:
hetero_test = sm.stats.diagnostic.het_goldfeldquandt
y = T40_complete['T40.4']
X = T40_complete.drop(columns=['T40.4', 'year', 'county_code'])
X_std = scale_df(X,scaler)
het_F_stat, het_p_stat, z = hetero_test(y, X_std)

#Check for multi-colinearity
vif = vif(X_std)
print(vif)

#modeling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
