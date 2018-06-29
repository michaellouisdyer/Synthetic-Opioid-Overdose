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
from tabulate import tabulate

def model_comparison(X,y,models):
    """Takes a list of LinearDataset models, as well as their X and y data and returns two dataframes, one describing their coefficients, and another describing their errors"""

    fig, axes = plt.subplots(1,len(models))
    coef_matrix = pd.DataFrame(index = X.columns)
    error_matrix = pd.DataFrame(index =['Train_R2','Test_R2', 'Test_RSS', 'Train_RSS', 'Unstandardized_Test_RSS', 'Unstandardized_Train_RSS'])

    for model in models:
        #initializes the models
        model.set_up()


        coef_matrix[model.name] = model.get_coefs()

        error_matrix[model.name] = model.test_and_train_errs()
    return coef_matrix, error_matrix

def all_plot_actual_predicted(models):
    fig, axes = plt.subplots(2,2)
    axes = axes.flatten()
    for i, model in enumerate(models):
        model.plot_actual_predicted(axes[i])
    plt.show()

    #main
mcd_main =  get_data()
T40 = drop_nulls(mcd_main,['T40.4'])
T40_complete = impute_df(T40, KNN(5))
y = T40_complete['T40.4']
X = T40_complete.drop(columns=['T40.4', 'year', 'county_code', 'T40.7', 'poverty_rate_native_american', 'poverty_rate_pacific_islander','college_degree', 'poverty_rate'])
# X = T40_complete.drop(columns=['T40.4', 'year','T40.7', 'poverty_rate_native_american', 'poverty_rate_pacific_islander','poverty_rate'])
# X = T40_complete[['T40.5','household_income', 'poverty_rate', 'unemployment_rate']]

l1_ratio = np.linspace(0.1,1,50)
cv=10
# alphas = [0.1,0.5,1,5,10]
alphas = np.linspace(0.1,100,50)
elastic = LinearDataset(X,y,ElasticNetCV(l1_ratio=l1_ratio), name='ElasticNet')

ridge = LinearDataset(X,y,RidgeCV(cv=cv, alphas =alphas), name='Ridge')
lasso = LinearDataset(X,y,LassoCV(cv=cv), name='Lasso')
linear = LinearDataset(X,y,LinearRegression(), name='linear')

models = [linear, elastic, ridge, lasso]

coef_matrix, error_matrix = model_comparison(X, y, models)
print('\n', coef_matrix)
print('\n', error_matrix)

# print(tabulate(coef_matrix.round(2), headers='keys', tablefmt='pipe'))
# print(tabulate(error_matrix.round(2), headers='keys', tablefmt='pipe'))
all_plot_actual_predicted(models)


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
