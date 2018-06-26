
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

from pandas.plotting import scatter_matrix
from EDA import impute_df, drop_nulls
mpl.rcParams.update({
    'font.size'           : 20.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'medium',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'medium',
})




#main

mcd, mcd_main, mcd_wide, mcd_2015, mcd_2016 =  get_data()
T40 = drop_nulls(mcd_main,['T40.4'])
T40_complete = impute_df(T40, SoftImpute())
