import pandas as pd

def read_data(filename, delim):
    df = pd.read_csv(filename, delimiter=delim)
    df.columns =  df.columns.str.lower().str.replace(' ', '_')
    return df

def main():
    mcd = read_data('data/MCD_2010_2016_T40.txt', delim='\t')
    mcd_2015 = mcd.query('year == 2015')

    return mcd
main()
