
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
from statsmodels.graphics.gofplots import qqplot
from scipy.stats.mstats import normaltest


mpl.rcParams.update({
    'font.size'           : 20.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'medium',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'medium',
})

def scale_df(df,scaler):
    """Input: df = Pandas Dataframe of features to be scaled
             Scaler object"""
    scaler.fit(df)
    return pd.DataFrame(data=scaler.transform(df), columns=df.columns, index=df.index)

def vif(X):
    """Returns the variance inflation factor for dataframe of features"""
    vif = pd.DataFrame()
    vif["VIF Factor"] = [c_vif(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns
    return vif

def get_coefs(model,X):
    """Returns model coefficients for sklearn models"""
    df = pd.DataFrame(model.coef_, index = X_train.columns)
    return df

def plot_MSE(model, X, ax=plt, c_title='', alphas = None):
    """Plots the mean squared error for various alphas in a cross-validation"""
    # if all(alphas) == None:
    alphas = model.alphas_
    m_log_alphas = -np.log10(alphas)
    ymin, ymax = model.mse_path_.min()*0.9, model.mse_path_.max()*1.1
    ax.plot(m_log_alphas, model.mse_path_, ':')
    # ax.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
    ax.axvline(-np.log10(model.alpha_), linestyle='--', color='k',label='alpha: CV estimate')

    ax.legend(list(range(1, model.mse_path_.shape[1]+1)), title='Fold')

    ax.xlabel('-log(alpha)')
    ax.ylabel('Mean square error')
    ax.title(c_title + ' Mean square error on each fold')
    ax.axis('tight')
    ax.ylim(ymin, ymax)

def plot_coeff_paths(model, X, y, ax=plt, c_title='', alphas = None):
    """Plots coefficient paths for various alphas in a cross-validation"""
    # if all(alphas) == None:
    alphas = model.alphas_
    m_log_alphas = np.log10(alphas)
    coeffs = model.path(X,y)[1]
    ymin, ymax = coeffs.min(), coeffs.max()
    ax.plot(m_log_alphas, coeffs.T)
    ax.legend(X.columns, title='Feature', loc='upper left', bbox_to_anchor=(1,1))
    ax.xlabel('log(alpha)')
    ax.title(c_title + ' Coefficient Descent')
    ax.ylim(ymin, ymax)

def test_and_train_errs(model, X_train, y_train, X_test, y_test):
    """Returns the residual sum of squares for training and test sets"""
    cv_error = {}
    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)
    cv_error['train'] = rss(y_train, y_hat_train)
    cv_error['test']  = rss(y_test, y_hat_test)
    return cv_error


def plot_actual_predicted(model, X, y, ax=plt):
    """Plots model predicted values verus actual values"""
    ax.scatter(model.predict(X)), y)
    model_name = model.__class__.__name__
    ax.set_title(model_name + ' Actual vs. predicted')

def rss(y, y_hat):
    """Returns the residual sum of sqaures"""
    return np.mean((y  - y_hat)**2)

def find_best_models(models, X_train, y_train, X_test, y_test):
    """Takes a list of fitted models, test and training data and returns matrices describing their coefficients and error statistics as well as their residuals"""
    fig, axes = plt.subplots(1,len(models))
    coef_matrix = pd.DataFrame(index = X_train.columns)
    error_matrix = pd.DataFrame(index =['Train_R2','Test_R2', 'Test_RSS', 'Train_RSS'])
    resid_matrix  = pd.DataFrame()
    for i, model in enumerate(models):
        model_name = model.__class__.__name__
        coef_matrix[model_name] = model.coef_
        train_r2 = model.score(X_train, y_train)
        test_r2 = model.score(X_test, y_test)
        plot_actual_predicted(model,X_test,y_test, ax=axes[i])
        rss_errors  = test_and_train_errs(model, X_train, y_train, X_test, y_test)
        error_matrix[model_name] = [train_r2, test_r2, rss_errors['train'], rss_errors['test']]
        resid_matrix[model_name] = y_train - model.predict(X_train)
    return coef_matrix, error_matrix, resid_matrix

def qqplots(resid_matrix):
    """Creates quantile-quantile plots bases on the matrix of residuals with each column representing a model"""
    fig, axes = plt.subplots(2,2)
    axes =  axes.flatten()
    for i, col in enumerate(resid_matrix.columns):
        qqplot(resid_matrix[col], ax=axes[i])
        axes[i].set_title(col)

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
X_std['constant'] = 1
het_F_stat, het_p_stat, z = hetero_test(y, X_std)

#Check for multi-colinearity
vif = vif(X_std)
print(vif)

#modeling
#create a holdout test set
X_train, X_test, y_train, y_test = train_test_split(X_std, np.log(y), test_size=0.20)

#initialize models
lasso_model = LassoCV(cv=10).fit(X_train, y_train)
ridge_alphas = np.linspace(1,10,100)
ridge_model = RidgeCV(alphas = ridge_alphas, cv=10).fit(X_train, y_train)
elastic_l1_ratio = np.linspace(0.1,1,11)
elastic_model = ElasticNetCV(l1_ratio=elastic_l1_ratio, cv=10).fit(X_train, y_train)
linear_model = LinearRegression().fit(X_train, y_train)
# Display results
# plot_MSE(lasso_model, X_train, c_title="Lasso")
# plot_coeff_paths(lasso_model, X_train, y_train, c_title='lasso')
# plot_MSE(ridge_model, X_train, c_title="Ridge", alphas = ridge_alphas)
# plot_coeff_paths(ridge_model, X_train, y_train, c_title='Ridge', alphas = ridge_alphas)
# plot_MSE(elastic_model, X_train, c_title="Elastic")
# plot_coeff_paths(elastic_model, X_train, y_train, c_title='Elastic')
# print(get_coefs(elastic_model, X_train))
models = [linear_model, lasso_model, ridge_model, elastic_model]
coef_matrix, error_matrix, resid_matrix = find_best_models(models, X_train, y_train, X_test, y_test)

# Visually check for normality of resiudals using qqplots
qqplots(resid_matrix)

# test for normality of residuals
normality_assumptions = [(col, normaltest(resid_matrix[col])) for col in resid_matrix]
plt.show()
