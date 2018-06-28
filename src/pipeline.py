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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from EDA import impute_df, drop_nulls
from statsmodels.stats.outliers_influence import variance_inflation_factor as c_vif
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LinearRegression
from sklearn.linear_model.coordinate_descent import LinearModelCV
from statsmodels.graphics.gofplots import qqplot
from scipy.stats.mstats import normaltest
from linear_models import LinearDataset

def model_comparison(X,y,models):
    fig, axes = plt.subplots(1,len(models))
    coef_matrix = pd.DataFrame(index = X.columns)
    error_matrix = pd.DataFrame(index =['Train_R2','Test_R2', 'Test_RSS', 'Train_RSS'])

    for model in models:
        model.set_up()
        coef_matrix[model.name] = model.get_coefs()
        error_matrix[model.name] = model.test_and_train_errs()
    return coef_matrix, error_matrix

    #main
mcd, mcd_main, mcd_wide, mcd_2015, mcd_2016 =  get_data()
T40 = drop_nulls(mcd_main,['T40.4'])
T40_complete = impute_df(T40, KNN(5))

y = T40_complete['T40.4']
X = T40_complete.drop(columns=['T40.4', 'year', 'county_code'])
# X = T40_complete[['T40.5','household_income', 'poverty_rate', 'unemployment_rate']]
l1_ratio = np.linspace(0.1,1,11)
elastic = LinearDataset(X,y,ElasticNetCV(l1_ratio=l1_ratio), name='ElasticNet')

ridge = LinearDataset(X,y,RidgeCV(), name='Ridge')
lasso = LinearDataset(X,y,LassoCV(), name='Lasso')

models = [elastic, ridge, lasso]

coef_matrix, error_matrix = model_comparison(X, y, models)
print('\n', coef_matrix)
print('\n', error_matrix)


# fig, ax = plt.subplots()
# # linear_test.plot_actual_predicted(ax=ax)
# # linear_test.plot_MSE(ax=ax)
# print(linear_test.find_residuals())
# linear_test.plot_qqplot()
# plt.show()
#
# ridge = LinearDataset(X,y,RidgeCV(alphas=[0,1]), name='Ridge')
# ridge.scale_X()
# ridge.add_constant()
# ridge.log_transform_y()
# ridge.test_split()
# ridge.fit_cross_val()
# RidgeCV().fit(X,y)
