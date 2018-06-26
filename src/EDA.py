### eda
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.plotly as py
import plotly.offline as plto
import plotly.figure_factory as ff
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from fancyimpute import SimpleFill, KNN, SoftImpute, IterativeSVD, MICE, MatrixFactorization, NuclearNormMinimization, BiScaler
from pandas.plotting import scatter_matrix
from prep import get_data, read_data
sns.set()
#change default font sizes
mpl.rcParams.update({
    'font.size'           : 20.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'medium',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'medium',
})

T40_dict = {'T40.1': 'Heroin',
    'T40.2': 'Other opioids (Morphine, Oxycodone)',
    'T40.3': 'Methadone',
    'T40.4': 'Synethetic Opioids (Fentanyl, Propoxyphene)',
    'T40.5': 'Cocaine',
    'T40.6': 'Other and unspecified narcotics',
    'T40.7': 'Cannabis (derivatives)'}

def plot_years():
    fig, ax =  plt.subplots()
    drug_X  = drugs_years.drop(columns=['county_code', 'population'])
    ax.plot(drug_X)
    ax.set_ylabel('Yearly Deaths')
    ax.set_title('Multiple Cause of Death by Drug Type, U.S.')
    ax.legend([T40_dict[col] for col in drug_X.columns], fontsize='medium')

def plot_counties(top_counties,col_name, title, legend_title):
    fig = ff.create_choropleth(fips=top_counties.index.values, values = top_counties[col_name],    binning_endpoints=list(np.linspace(0.0,top_counties[col_name].max(),11)),     county_outline={'color': 'rgb(255,255,255)', 'width': 0.5}, round_legend_values=False, title=title, legend_title=legend_title)
    plto.plot(fig, filename=title.lower().replace(" ", "_")+'.html')


def ANOVA(df):
    formula = 'deaths ~ C(multiple_cause_of_death_code) * year'
    model = ols(formula, df).fit()
    aov_table = anova_lm(model, typ=2)
    return aov_table

def drop_nulls(df, cols):
    for col in cols:
        df = df[pd.notnull(df[col])]
    return df

def impute_df(df,algorithm):
    return pd.DataFrame(data=algorithm.complete(df), columns=df.columns, index=df.index)

def determine_impute(df):
    algorithms = [SimpleFill, KNN, SoftImpute, IterativeSVD,  MatrixFactorization]
    MSE = {}
    for alg in algorithms:
        print(alg)
        X_complete = impute_df(df, alg())
        alg_mse = ((X_complete[pd.isnull(df)]) ** 2).sum().mean()
        print(alg.__name__, alg_mse)
        MSE[alg.__name__] =  alg_mse
    return MSE

mcd, mcd_main, mcd_wide, mcd_2015, mcd_2016 =  get_data()
county_dict = dict(mcd[['county_code','county']].drop_duplicates().values)

def find_top_counties():
    county_df = mcd[['county_code','county']].drop_duplicates().set_index('county_code')
    drugs_years = mcd_wide.groupby('year').sum()
    top_counties = mcd_wide
    top_counties['total_deaths'] = top_counties[list(T40_dict.keys())].sum(axis=1)
    top_counties = top_counties.groupby('county_code').sum()
    top_counties['death_ratio'] =  1000* top_counties['total_deaths']/top_counties['population']
    top_counties['synthetic_ratio'] =  1000* top_counties['T40.4']/top_counties['population']

    ##plot counties
    # top_counties = top_counties.sort_values('death_ratio', ascending = False)
    # plot_counties(top_counties,'synthetic_ratio', title="U.S. Counties by Mutliple Cause of Death Involving Synthetic Opioids 2010 to 2016", legend_title='Deaths per 1000 Residents')
    # plot_counties(top_counties,'death_ratio', title="U.S. Counties by Mutliple Cause of Death Involving Drugs 2010 to 2016", legend_title='Deaths per 1000 Residents')

#ANOVA for cocaine, synthetic
drugs_test = mcd.groupby(by=['year','multiple_cause_of_death_code']).sum()['deaths'].reset_index().query('multiple_cause_of_death_code in ["T40.4" ,  "T40.5"]')
# print(ANOVA(drugs_test))

#top counties
top_counties = top_counties.join(county_df)
print(top_counties[['death_ratio', 'county']].sort_values('death_ratio', ascending=False).head())
print(top_counties[['synthetic_ratio', 'county']].sort_values('synthetic_ratio', ascending=False).head())

#dataframe with T40 deaths as the target
T40 = drop_nulls(mcd_main,['T40.4'])

# iterate through various imputation techniques to
# MSE_dict = determine_impute(T40)

#used softimpute to find missing data points
# T40_complete = impute_df(T40, SoftImpute())

#correlations
mcd_wide['death_ratio'] =  1000* mcd_wide['total_deaths']/mcd_wide['population']
mcd_wide['synthetic_ratio'] =  1000* mcd_wide['T40.4']/mcd_wide['population']

#plot T40 Correlations
# sns.pairplot(mcd_wide[list(T40_dict.keys())].fillna(0))
# plot t40 and t50 with economic
# sns.pairplot(mcd_main[['poverty_rate', 'unemployment_rate']].fillna(0))
census =mcd_main[['poverty_rate', 'unemployment_rate','household_income']]
census['household_income_log'] = np.log(census.pop('household_income'))
census_std = (census - census.mean()) / census.std()
sns.pairplot(census_std)
# census.plot.scatter('poverty_rate', 'unemployment_rate')
# census.plot.scatter('household_income', 'unemployment_rate')
# census.plot.scatter('poverty_rate', 'household_income')
plt.show()
