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
from prep import get_data, read_data
# dfs = get_data()

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

mcd, mcd_main, mcd_wide, mcd_2015, mcd_2016 =  get_data()
county_dict = dict(mcd[['county_code','county']].drop_duplicates().values)

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
