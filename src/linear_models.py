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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas.plotting import scatter_matrix
from EDA import impute_df, drop_nulls
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LinearRegression
from sklearn.linear_model.coordinate_descent import LinearModelCV
from statsmodels.graphics.gofplots import qqplot
from scipy.stats.mstats import normaltest
from utils import XyScaler


class LinearDataset:
    """Adds functionalities to linear_model"""

    def __init__(self, X, y, linear_modelcv, name):
        self.X = X.copy()
        self.X_std = None
        self.y_std = None
        self.y = y.copy()
        self.model_cv = linear_modelcv
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.type = self.model_cv.__class__.__name__
        self.name = self.type
        self.hyperparameters = pd.Series()

        self.scaler = XyScaler()

    def scale_X(self):
        """Scales X and y"""
        self.scaler.fit(self.X, self.y)
        X_std, y_std = self.scaler.transform(self.X, self.y)
        self.X_std = pd.DataFrame(data=X_std, columns=self.X.columns, index=self.X.index)
        self.y_std = pd.Series(data=y_std)

    def add_constant(self):
        self.X['constant'] = 1

    def goldfeldtquandt(self):
        """Conducts a Goldfeldt-Quandt test for heteroscedasticity with the null hypothesis that errors are normally distributed""""
        het_F_stat, het_p_stat, z = sm.stats.diagnostic.het_goldfeldquandt(self.y, self.X)
        return {"F": het_F_stat, "p": het_p_stat}

    def vif(self):
        """Returns the variance inflation factor for dataframe of features to test for multicolinearity, scores of 5 and up indicate multicolinearity"""
        vif = pd.DataFrame()
        vif["Features"] = self.X_std.columns
        vif["VIF Factor"] = [variance_inflation_factor(
            self.X_std.values, i) for i in range(self.X_std.shape[1])]
        return vif

    def test_split(self, ratio=0.25):
        """Splits a training set with a given ratio""""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_std, self.y_std, test_size=ratio, random_state=8)

    def log_transform_y(self):
        """Log transforms the target vector"""
        self.y = np.log(self.y)

    def get_coefs(self):
        """Returns model coefficients for sklearn models"""
        df = pd.DataFrame(self.model_cv.coef_, index=self.X_train.columns)
        return df

    def fit_cross_val(self, cv=10, alphas=None, l1_ratio=None):
        """Fits sklearn LinearModelCV and adds hyperparameters to object name"""
        self.model_cv.fit(self.X_train, self.y_train)
        if self.name in ['ElasticNetCV', 'RidgeCV', 'LassoCV']:
            self.hyperparameters['a'] = self.model_cv.alpha_
        if self.name == 'ElasticNetCV':
            self.hyperparameters['l1_ratio'] = self.model_cv.l1_ratio_
        self.name += str(dict(self.hyperparameters.round(3)))

    def plot_MSE(self, ax=plt, c_title=''):
        """Plots the mean squared error for various alphas in a cross-validation"""
        alphas = self.model_cv.alphas_
        m_log_alphas = -np.log10(alphas)
        ymin, ymax = self.model_cv.mse_path_.min()*0.9, self.model_cv.mse_path_.max()*1.1
        ax.plot(m_log_alphas, self.model_cv.mse_path_, ':')
        ax.axvline(-np.log10(self.model_cv.alpha_), linestyle='--',
                   color='k', label='alpha: CV estimate')

        ax.legend(list(range(1, self.model_cv.mse_path_.shape[1]+1)), title='Fold')

        ax.set_xlabel('-log(alpha)')
        ax.set_ylabel('Mean square error')
        ax.set_title(c_title + ' Mean square error on each fold')
        ax.set_ylim(ymin, ymax)

    def plot_coeff_paths(self, ax=plt, c_title=''):
        """Plots coefficient paths for various alphas in a cross-validation"""
        alphas = self.model_cv.alphas_
        m_log_alphas = np.log10(alphas)
        coeffs = self.model_cv.path(self.X_train, y_train)[1]
        ymin, ymax = coeffs.min(), coeffs.max()
        ax.plot(m_log_alphas, coeffs.T)
        ax.legend(self.X_train.columns, title='Feature', loc='upper left', bbox_to_anchor=(1, 1))
        ax._set_xlabel('log(alpha)')
        ax.set_title(c_title + ' Coefficient Descent')
        ax.set_ylim(ymin, ymax)

    def plot_actual_predicted(self,  ax=plt, y_log=True):
        """Plots model predicted values verus actual values"""
        y_hat_test = self.y_hat_test
        y_test = self.y_test

        if y_log:
            y_test = np.log(self.y_test)
            y_hat_test = np.log(self.y_hat_test)

        ax.scatter(y_test,  y_hat_test)

        model_name = self.model_cv.__class__.__name__
        ax.set_title(model_name + ' Actual vs. Predicted')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        plt.subplots_adjust(hspace=.300, wspace=.200)

    def _rss(self, y, y_hat):
        """Returns the residual sum of sqaures"""
        return np.mean((y - y_hat)**2)

    def predict(self):
        """Returns standardized and unstandardized predictions"""

        self.y_hat_train = self.model_cv.predict(self.X_train)
        self.y_hat_test = self.model_cv.predict(self.X_test)

        self.X_train_unstandardized, self.y_hat_train_unstandardized = self.scaler.inverse_transform(
            self.X_train, self.y_hat_train)

        self.X_test_unstandardized, self.y_hat_test_unstandardized = self.scaler.inverse_transform(
            self.X_test, self.y_hat_test)

        self.X_test_unstandardized, self.y_test_unstandardized = self.scaler.inverse_transform(
            self.X_test, self.y_test)

    def test_and_train_errs(self):
        """Returns the residual sum of squares for training and test sets"""

        rss_train = self._rss(self.y_train, self.y_hat_train)
        rss_test = self._rss(self.y_test, self.y_hat_test)

        rss_train_unstandardized = self._rss(self.y_train, self.y_hat_train_unstandardized)
        rss_test_unstandardized = self._rss(self.y_test, self.y_hat_test_unstandardized)

        r2_train = self.model_cv.score(self.X_train, self.y_train)
        r2_test = self.model_cv.score(self.X_test, self.y_test)

        return [r2_train, r2_test, rss_train, rss_test, rss_train_unstandardized, rss_test_unstandardized]

    def find_residuals(self):
        """Returns difference of actual and predicted targets for training""""
            return self.y_train - self.model_cv.predict(self.X_train)

    def plot_qqplot(self):
        """Creates quantile-quantile plots bases on the residuals"""
        qqplot(self.find_residuals())

    def set_up(self, y_log=True, ratio=0.35):
        """Automates processing steps"""
        if y_log:
            self.log_transform_y()
        self.scale_X()
        self.add_constant()
        self.test_split(ratio=ratio)
        self.fit_cross_val()
        self.predict()
