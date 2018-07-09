import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from prep import get_data
from fancyimpute import KNN
from EDA import impute_df, drop_nulls
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LinearRegression
from LinearDataset import LinearDataset
from tabulate import tabulate
# Optional imports for testing assumptions
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor as c_vif
from scipy.stats.mstats import normaltest


def model_comparison(X, y, models):
    """Takes a list of LinearDataset models, as well as their X and y data and returns two dataframes, one describing their coefficients, and another describing their errors"""

    coef_matrix = pd.DataFrame(index=X.columns)
    error_matrix = pd.DataFrame(index=['Train_R2', 'Test_R2', 'Test_RSS',
                                       'Train_RSS', 'Unstandardized_Test_RSS', 'Unstandardized_Train_RSS'])

    for model in models:
        # initializes the models
        model.set_up()

        coef_matrix[model.name] = model.get_coefs()

        error_matrix[model.name] = model.test_and_train_errs()
    return coef_matrix, error_matrix


def all_plot_actual_predicted(models):
    """Plots actual vs. predicted values for all models"""
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    for i, model in enumerate(models):
        model.plot_actual_predicted(axes[i])
    plt.show()


def main():

    # Initialize data and set X and y
    mcd_main = get_data()
    T40 = drop_nulls(mcd_main, ['T40.4'])
    T40_complete = impute_df(T40, KNN(5))
    y = T40_complete['T40.4']
    X = T40_complete.drop(columns=['T40.4', 'year', 'county_code', 'T40.7', 'poverty_rate_native_american',
                                   'poverty_rate_pacific_islander', 'college_degree', 'poverty_rate'])

    # Create regression models for comparison
    l1_ratio = np.linspace(0.1, 1, 100)

    cv = 5  # Number of k-fold cross validations

    alphas = np.linspace(0.1, 100, 100)
    elastic = LinearDataset(X, y, ElasticNetCV(l1_ratio=l1_ratio), name='ElasticNet')

    ridge = LinearDataset(X, y, RidgeCV(cv=cv, alphas=alphas), name='Ridge')
    lasso = LinearDataset(X, y, LassoCV(cv=cv), name='Lasso')
    linear = LinearDataset(X, y, LinearRegression(), name='linear')

    models = [linear, elastic, ridge, lasso]

    # Compare models
    coef_matrix, error_matrix = model_comparison(X, y, models)

    print(tabulate(coef_matrix.round(2), headers='keys', tablefmt='pipe'))
    print(tabulate(error_matrix.round(2), headers='keys', tablefmt='pipe'))
    all_plot_actual_predicted(models)

    # Plot coefficient path for selected model
    fig, ax = plt.subplots()
    lasso.plot_coeff_paths(ax=ax, c_title='Lasso ')
    plt.show()


if __name__ == '__main__':
    main()
