import pandas as pd

def read_data(filename, delim, encoding='utf-8', skiprow=0):
    df = pd.read_csv(filename, delimiter=delim, encoding=encoding, skiprows=skiprow)
    df.columns =  df.columns.str.lower().str.replace(' ', '_')
    return df

def census_data():

    delim = ','
    encode ='ISO-8859-1'

    df_poverty_15 = read_data('data/acs_data/ACS_15_1YR_S1701_with_ann.csv', delim = delim, encoding=encode, skiprow=1)
    df_poverty_16 = read_data('data/acs_data/ACS_16_1YR_S1701_with_ann.csv', delim = delim, encoding=encode, skiprow=1)
    df_poverty_15['year'] = 2015.0
    df_poverty_16['year'] = 2016.0
    df_poverty = df_poverty_15.append(df_poverty_16)
    df_poverty = df_poverty[['year','id2', 'percent_below_poverty_level;_estimate;_population_for_whom_poverty_status_is_determined']]
    df_poverty.columns = ['year','county_code', 'poverty_rate']

    df_income_15 = read_data('data/acs_data/ACS_15_1YR_S1901_with_ann.csv', delim = delim, encoding=encode, skiprow=1)
    df_income_16 = read_data('data/acs_data/ACS_16_1YR_S1901_with_ann.csv', delim = delim, encoding=encode, skiprow=1)
    df_income_15['year'] = 2015.0
    df_income_16['year'] = 2016.0
    df_income = df_income_15.append(df_income_16)
    df_income = df_income[['year','id2', 'households;_estimate;_total']]
    df_income.columns = ['year','county_code', 'household_income']

    df_employment_15 = read_data('data/acs_data/ACS_15_1YR_S2301_with_ann.csv', delim = delim, encoding=encode, skiprow=1)
    df_employment_16 = read_data('data/acs_data/ACS_16_1YR_S2301_with_ann.csv', delim = delim, encoding=encode, skiprow=1)
    df_employment_15['year'] = 2015.0
    df_employment_16['year'] = 2016.0
    df_employment = df_employment_15.append(df_employment_16)
    df_employment = df_employment[['year','id2', 'unemployment_rate;_estimate;_population_16_years_and_over']]
    df_employment.columns = ['year','county_code', 'unemployment_rate']

    return df_income.merge(df_poverty, on=['year','county_code']).merge(df_employment, on=['year','county_code'])

def get_data():
    mcd = read_data('data/MCD_2010_2016_T40.txt', delim='\t')
    mcd = mcd[pd.notnull(mcd['year'])] #remove footer
    mcd_pivot = mcd.pivot_table(index=['year','county_code'], columns='multiple_cause_of_death_code', values='deaths').reset_index() #wide to long
    pop_df = mcd[['year','county_code', 'population']].drop_duplicates() #find population for each county
    mcd_wide = pd.merge(pop_df, mcd_pivot, on =['year','county_code']) # add the two together
    census = census_data()
    mcd_main =  pd.merge(mcd_wide, census, on=['year','county_code'])
    mcd_2015 = mcd_main.query('year == 2015') #subset for 2015
    mcd_2016 = mcd_main.query('year == 2016') #subset for 2015
    return mcd_wide, mcd_2015, mcd_2016

    ### eda
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.plotly as py
import plotly.offline as plto
import plotly.figure_factory as ff
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
from prep import get_data
mcd_wide, mcd_2015, mcd_2016 =  get_data()
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

drugs_years = mcd_wide.groupby('year').sum()
top_counties = mcd_wide
top_counties['total_deaths'] = top_counties[list(T40_dict.keys())].sum(axis=1)
top_counties = top_counties.groupby('county_code').sum()
top_counties['death_ratio'] =  1000* top_counties['total_deaths']/top_counties['population']
top_counties['synthetic_ratio'] =  1000* top_counties['T40.4']/top_counties['population']
top_counties = top_counties.sort_values('death_ratio', ascending = False)
plot_counties(top_counties,'synthetic_ratio', title="U.S. Counties by Mutliple Cause of Death Involving Synthetic Opioids 2010 to 2016", legend_title='Deaths per 1000 Residents')
plot_counties(top_counties,'death_ratio', title="U.S. Counties by Mutliple Cause of Death Involving Drugs 2010 to 2016", legend_title='Deaths per 1000 Residents')


# counties_years = mcd_wide.groupby('county')
# plot_years()
# plt.show()
