# overdose_CDC

## Data
Overdose Data was gathered from the CDC WONDER Database using the T40 (Poisoning by narcotics and psychodysleptics) classifier for Multiple Cause of Death, by county, year(2010-2016) and drug  
Economic Data was gathered from the Census Bureau American Community Surveys by county and year(2015-2016)
Facts included are Unemployment Rate for Adults 16+, Total Household Income Estimate, and Percent Below Poverty Level

## EDA
Plotting deaths over the years showed a trend where all drug deaths increased from 2013 to 2016, with a notable increase in synthetic opioids. In addition, both fentanyl and cocaine had a large increase in deaths from 2015 to 2016.
<img src='images/drug_by_year.png' width=900>
County analysis showed that the highest overall MCD drug deaths were in Kentucky and West Virginia, while Washington County was in the top five for both overall and synthetic deaths.

| county_code|            death_ratio  |               county|
| ---------  |----------------------- | --------------------- |
|21197.0|         1.030683|      Powell County, KY|
|54047.0|         0.910671|    McDowell County, WV|
|23029.0|         0.890302|  Washington County, ME|
|54109.0|         0.818653|     Wyoming County, WV|
|54005.0|         0.793864|       Boone County, WV|

| county_code|            synthetic_ratio  |               county|
| ---------  |----------------------- | --------------------- |
|21097.0|             0.589939|    Harrison County, KY|
|23029.0|             0.476948|  Washington County, ME|
|21081.0|             0.401236|       Grant County, KY|
|24001.0|             0.346596|    Allegany County, MD|
|39027.0|             0.310137|     Clinton County, OH|


A scatter matrix comparing distributions for T40 MCD codes signified linear relationships between many of the codes; this is expected as each category is not exclusive and an autopsy can often list more than one drug as an MCD. Notable relationships are seen between T40.2 (Methadone ) and T40.3(Other Opioids), and T40.4 (Synthetic Opioids) and T40.5 (Cocaine).

<img src="images/T40_scatter.png" width=800>

For census data, a scatter matrix revealed a relationship between poverty rate and unemployment_rate, as well as a relationship of those two with the log of households income(taken because income was not normally distributed)

<img src="images/census_scatter_matrix.png" width=800>

## Hypothesis Testing
A two-way analysis of variance _factors(year, drug(cocaine, synthetic opioids))_
df showed that while there was not a significant difference for the mean number of deaths for each drug across all years (_F_=0.07, _p_=0.79), there was a significant interaction between cocaine and synthetic deaths (_F_ = 3.75, _p_=0.08), where deaths by each drug increased by a function of each other eash subsequent year. There was a significant effect of year(_F_ = 15.73, _p_<0.01), where deaths increased from 2010 to 2016.


## Modeling
A subset of the data consisting of all counties with recorded non-zero synthetic opioid deaths (T40.4) in 2015 and 2016 was chosen to model on. The target consisted of each observation of T40.4 deaths, and the features were selected from the remaining drug codes (T40.1 - T40.3, T40.5 - T40.7) as well as county population, household income, and unemployment and poverty rates.
Because county X year observations were missing values, a comparison of various imputation methods resulted in a K-nearest neighbors (_k_ = 5) as the most best fit for the data. 2,960 missing values were imputed using this technique.
The data was then standardized and tested for homoscedasticity using Goldfeld-Quandt (_F_: 1.08, _p_:0.23).

Multicolinearity was tested by computing the Variance Inflation Factor for each feature, given by the following table:  


| Features                      |   VIF Factor |
|:------------------------------|-------------:|
| population                    |        2.553 |
| T40.1                         |        2.963 |
| T40.2                         |        2.912 |
| T40.3                         |        2.136 |
| T40.5                         |        3.033 |
| T40.6                         |        1.231 |
| household_income              |        2.836 |
| low_income_families           |        2.873 |
| poverty_rate_white            |        2.831 |
| poverty_rate_african_american |        1.777 |
| poverty_rate_asian            |        1.344 |
| poverty_rate_hispanic         |        1.771 |
| unemployment_rate             |        1.489 |
| disability_employed           |        2.761 |

Cross validation of hyperparameters (n_folds = 10) of ridge, lasso and elastic net models was run and fit scores were computed for each and compared to the standard linear model

### Coefficients


Including data from 2013- 2016
#### Table of coefficients by model

|                               |   LinearRegression{} |   ElasticNetCV{'a': 0.048, 'l1_ratio': 0.1} |   RidgeCV{'a': 32.72} |   LassoCV{'a': 0.011} |
|:------------------------------|---------------------:|--------------------------------------------:|----------------------:|----------------------:|
| population                    |               -0.446 |                                     ** -0.36 ** |                -0.35  |                -0.391 |
| T40.1                         |                0.236 |                                     **  0.242** |                 0.247 |                 0.231 |
| T40.2                         |                0.218 |                                     **  0.177** |                 0.173 |                 0.189 |
| T40.3                         |                0.025 |                                     **  0.024** |                 0.029 |                 0.014 |
| T40.5                         |                0.507 |                                     **  0.456** |                 0.439 |                 0.493 |
| T40.6                         |                0.005 |                                     **  0.006** |                 0.009 |                 0.003 |
| household_income              |                0.089 |                                     **  0.057** |                 0.066 |                 0.053 |
| low_income_families           |               -0.001 |                                     ** -0.003** |                -0.013 |                -0     |
| poverty_rate_white            |                0.028 |                                     **  0    ** |                 0.014 |                 0     |
| poverty_rate_african_american |               -0.063 |                                     ** -0.039** |                -0.045 |                -0.034 |
| poverty_rate_asian            |                0.091 |                                     **  0.083** |                 0.087 |                 0.077 |
| poverty_rate_hispanic         |                0.037 |                                     **  0.018** |                 0.027 |                 0.005 |
| unemployment_rate             |               -0.15  |                                     ** -0.136** |                -0.14  |                -0.133 |
| disability_employed           |               -0.152 |                                     ** -0.134** |                -0.13  |                -0.136 |
