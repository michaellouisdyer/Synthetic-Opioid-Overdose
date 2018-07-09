import pandas as pd
import numpy as np


def read_data(filename, delim, encoding='utf-8', skiprow=0):
    df = pd.read_csv(filename, delimiter=delim, encoding=encoding, skiprows=skiprow)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df


def census_data():
    """Takes 1yr estimate files from the Census American Community Survey and returns a merged dataframe with metrics by county and year"""

    delim = ','
    encode = 'ISO-8859-1'

    years = [10, 11, 12, 13, 14, 15, 16]
    poverty = []
    income = []
    employment = []

    # add data for each year and concatenate
    for year in years:
        filename_poverty = 'data/acs_data/ACS_'+str(year)+'_1YR_S1701_with_ann.csv'
        df_poverty_year = read_data(filename_poverty, delim=delim, encoding=encode, skiprow=1)
        df_poverty_year['year'] = year + 2000
        poverty.append(df_poverty_year)

        filename_income = 'data/acs_data/ACS_'+str(year)+'_1YR_S1901_with_ann.csv'
        df_income_year = read_data(filename_income, delim=delim, encoding=encode, skiprow=1)
        df_income_year['year'] = year + 2000
        income.append(df_income_year)

        filename_employment = 'data/acs_data/ACS_'+str(year)+'_1YR_S2301_with_ann.csv'
        df_employment_year = read_data(filename_employment, delim=delim, encoding=encode, skiprow=1)
        df_employment_year['year'] = year + 2000
        employment.append(df_employment_year)

    df_poverty = pd.concat(poverty)
    df_income = pd.concat(income)
    df_employment = pd.concat(employment)

    # select features
    df_poverty = df_poverty[['year', 'id2',
                             'percent_below_poverty_level;_estimate;_population_for_whom_poverty_status_is_determined',
                             'percent_below_poverty_level;_estimate;_white_alone,_not_hispanic_or_latino', 'percent_below_poverty_level;_estimate;_race_and_hispanic_or_latino_origin_-_black_or_african_american_alone',
                             'percent_below_poverty_level;_estimate;_race_and_hispanic_or_latino_origin_-_american_indian_and_alaska_native_alone', 'percent_below_poverty_level;_estimate;_race_and_hispanic_or_latino_origin_-_asian_alone', 'percent_below_poverty_level;_estimate;_race_and_hispanic_or_latino_origin_-_native_hawaiian_and_other_pacific_islander_alone',
                             'percent_below_poverty_level;_estimate;_hispanic_or_latino_origin_(of_any_race)', "total;_estimate;_educational_attainment_-_bachelor's_degree_or_higher"]]
    df_poverty.columns = ['year', 'county_code', 'poverty_rate', 'poverty_rate_white', 'poverty_rate_african_american',
                          'poverty_rate_native_american', 'poverty_rate_asian', 'poverty_rate_pacific_islander', 'poverty_rate_hispanic', 'college_degree']

    df_income = df_income[[
        'year', 'id2', 'households;_estimate;_mean_income_(dollars)', 'families;_estimate;_$10,000_to_$14,999']]
    df_income.columns = ['year', 'county_code', 'household_income', 'low_income_families']

    df_employment = df_employment[['year', 'id2', 'unemployment_rate;_estimate;_population_16_years_and_over',
                                   'employed;_estimate;_disability_status_-_with_any_disability']]
    df_employment.columns = ['year', 'county_code', 'unemployment_rate', 'disability_employed']

    # merge dataframes
    census_data = df_income.merge(df_poverty, on=['year', 'county_code']).merge(
        df_employment, on=['year', 'county_code'])
    census_data = census_data.apply(pd.to_numeric, errors='coerce')

    return census_data


def get_all_data():
    """Reads and combines census and drug data """
    mcd = read_data('data/MCD_2010_2016_T40.txt', delim='\t')

    # remove footer
    mcd = mcd[pd.notnull(mcd['year'])]

    # convert wide to long
    mcd_pivot = mcd.pivot_table(
        index=['year', 'county_code'], columns='multiple_cause_of_death_code', values='deaths').reset_index()

    # find population for each county
    pop_df = mcd[['year', 'county_code', 'population']].drop_duplicates()

    # add the two together
    mcd_wide = pd.merge(pop_df, mcd_pivot, on=['year', 'county_code'])

    census = census_data()

    mcd_main = pd.merge(mcd_wide, census, on=['year', 'county_code'])

    # normalize education fields
    for col in ['college_degree', 'low_income_families', 'disability_employed']:
        mcd_main[col] = mcd_main[col]/mcd_main['population']

    return mcd, mcd_main, mcd_wide


def get_data():
    """Returns a subset of data for 2012 and on"""
    mcd, mcd_main, mcd_wide = get_all_data()
    return mcd_main.query('year > 2012')
