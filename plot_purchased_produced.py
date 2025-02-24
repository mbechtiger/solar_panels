#!/opt/local/bin/python

'''
plot solar panels production and Engie usage in kW per day
data made available in CSV - production by Sunny Boy converter and consumption/production by Enedis (Linky) web platforms
or imported from the sqlite database where the clean csv dataframes are stored
usage : ./plot_purchased_produces.py {--source=csv|sql} {--trace} {--plot}
created : 20220714 - marcelbechtiger@gmail.com
updated : 20221006, 20230117,
20231112 (python 3.10 and pandas>=2.0.0),
20240813 (added Sqlite, plotly; cleanup) - note that matplotlib displays some crap at the right of the plot with multiple inputs
20241222 pandas, matplotlib, plotly : must install specifically on Wilma/Python 3 using Software Manager
   ENEDIS changed files naming and CSV to XLSX : now concatenate CSV & XLSX dataframes
   removed matplotlib support to clean up code
'''

import numpy as np
import pandas as pd
from datetime import datetime as dt
from pathlib import Path
import sqlite3
import plotly.graph_objects as go
import sys
import argparse

BASE = '/Users/marcel/Projects/'
CSV_XLSX_FOLDER = BASE + 'SOLAR POWER METRICS'
SQLITE_DATABASE = BASE + 'python/solarPanels/solarPanels_data.sqlite'
SOLAR_LIMIT_MAX = 100000000

csvFileCount = 0

parser = argparse.ArgumentParser(prog='plot_purchased_produced.py',
    description='''
    Process solar panels (production) and Enedis/Linky (consumption/production)
    energy data. Version 20241222.
    ''',
    epilog='''
    The data is initially loaded from various CSV/XLSX,
    then stored in a Sqlite database for later & faster retrieval.
    Plotting uses plotly.
    Please note that the CSV/XLSX source folder and Sqlite database are hard-coded at the top of this script.
    ''')
parser.add_argument('-s', '--source', help='data source can be set to or SQL - default XLS',
    choices=['SQL', 'sql', 'XLS', 'xls'], default='XLS')
parser.add_argument('-t', '--trace', help='specify to activate tracing',
    action='store_true')
parser.add_argument('-p', '--plot', help='specify to activate plotting',
    action='store_true')
parser.add_argument('-i', '--info', help='specify to just get SQL stored info',
    action='store_true')
args = parser.parse_args()

source = args.source.lower()
trace = args.trace
plot = args.plot
info = args.info

if not info:
    print('*****importing data from %s with trace:%s and plot:%s' % (source.upper(), trace, plot))

# -------------------- dataframe pretty print --------------------

def dataframePrettyPrint(df):
    with pd.option_context(
        'display.max_rows', None,
        'display.max_columns', None,
        'display.precision', 3):
        print(df)
    print("*****%d rows in dataframe" % len(df))

# -------------------- consumed electricity Engie/Linky --------------------

def loadEnedisConsumedCsv(file, dummy):
    df = pd.read_csv(file, sep=';', skiprows=2, skipfooter=0, engine='python', on_bad_lines='skip')

    # ignore crappy rows
    df = df[df['Type de releve'] == 'Arrêté quotidien']

    dtm = lambda x: dt.strptime(x, '%Y-%m-%d')
    df['Horodate'] = df['Horodate'].str[:10].apply(dtm)
    df['Horodate'] = pd.to_datetime(df['Horodate'], format='%Y-%m-%d')
    df['Horodate'] = df['Horodate'].dt.strftime('%Y-%m-%d')

    # keep useable columns
    df_f = df.iloc[:, [0,2,3,16]]

    # normalize column names
    df_f = df_f.rename(columns={'Horodate': 'Date'})
    df_f = df_f.rename(columns={'EAS T': 'Total'})

    df_f_shift = df_f.Total.shift(1)
    df_f['Diff'] = df_f['Total'] - df_f_shift
    df_f['Diff'] = df_f['Diff'] / 1000
    df_f['Diff2']  = df_f['Diff'].shift(-1)

    ##dataframePrettyPrint(df_f)
    df_f = df_f.dropna(subset=['Diff2'])
    df_f = df_f.drop(['EAS F1', 'EAS F2', 'Total', 'Diff'], axis=1)
    df_f = df_f.rename(columns={'Diff2': 'Diff'})
    ##dataframePrettyPrint(df_f)

    return df_f

# -------------------- PV panels produced electricity SunnyBoy --------------------

def loadSunnyBoyCsv(file, dummy):
    df = pd.read_csv(file, sep=',', skiprows=8, skipfooter=0, engine='python', on_bad_lines='skip')

    dtm = lambda x: dt.strptime(x, '%d.%m.%Y')
    df['DD.MM.YYYY hh:mm:ss'] = df['DD.MM.YYYY hh:mm:ss'].str[:10].apply(dtm)
    df['DD.MM.YYYY hh:mm:ss'] = pd.to_datetime(df['DD.MM.YYYY hh:mm:ss'], format='%Y-%m-%d')
    df['DD.MM.YYYY hh:mm:ss'] = df['DD.MM.YYYY hh:mm:ss'] - pd.Timedelta(days=1)
    df['DD.MM.YYYY hh:mm:ss'] = df['DD.MM.YYYY hh:mm:ss'].dt.strftime('%Y-%m-%d')

    # keep useable columns
    df_f = df.iloc[:, [0,1]]

    # normalize column names
    df_f = df_f.rename(columns={'DD.MM.YYYY hh:mm:ss': 'Date'})
    df_f = df_f.rename(columns={'[Wh]': 'Total'})

    # ignore crappy values : 2022<date<2025, 0<wh<SOLAR_LIMIT_MAX
    df_f = df_f[df_f['Date'] >= '2022-01-01 00:00:00']
    df_f = df_f[df_f['Date'] <= '2025-01-01 00:00:00']
    df_f = df_f[df_f['Total'] < SOLAR_LIMIT_MAX]
    df_f = df_f[df_f['Total'] > 0.0]

    df_f_shift = df_f.Total.shift(1)
    df_f['Diff'] = df_f['Total'] - df_f_shift
    df_f['Diff'] = df_f['Diff'] / 1000

    ##dataframePrettyPrint(df_f)
    df_f = df_f.dropna(subset=['Diff'])
    df_f = df_f.drop(['Total'], axis=1)
    ##dataframePrettyPrint(df_f)

    return df_f

# -------------------- produced electricity Engie/Linky --------------------

def loadEnedisProducedCsv(file, dummy):
    df = pd.read_csv(file, sep=';', skiprows=2, skipfooter=0, engine='python', on_bad_lines='skip')

    # ignore crappy rows
    df = df[df['Type de releve'] == 'Arrêté quotidien']

    dtm = lambda x: dt.strptime(x, '%Y-%m-%d')
    df['Horodate'] = df['Horodate'].str[:10].apply(dtm)
    df['Horodate'] = pd.to_datetime(df['Horodate'], format='%Y-%m-%d')
    df['Horodate'] = df['Horodate'].dt.strftime('%Y-%m-%d')

    # keep useable columns
    df_f = df.iloc[:, [0,17]]

    # normalize column names
    df_f = df_f.rename(columns={'Horodate': 'Date'})
    df_f = df_f.rename(columns={'EAI T': 'Total'})

    df_f_shift = df_f.Total.shift(1)
    df_f['Diff'] = df_f['Total'] - df_f_shift
    df_f['Diff'] = df_f['Diff'] / 1000
    df_f['Diff2']  = df_f['Diff'].shift(-1)

    ##dataframePrettyPrint(df_f)
    df_f = df_f.dropna(subset=['Diff2'])
    df_f = df_f.drop(['Total', 'Diff'], axis=1)
    df_f = df_f.rename(columns={'Diff2': 'Diff'})
    ##dataframePrettyPrint(df_f)

    return df_f

# -------------------- consumed or produced electricity Engie/Linky --------------------

def loadEnedisXlsx(file, sheetName):
    # sheetname or index can be used
    df = pd.read_excel(file, sheet_name=sheetName, usecols='B:C', names=['Date','Diff'],
        skiprows=13, parse_dates=[0], date_format='%d/%m/%Y',
        decimal=',', dtype={'Diff':float})

    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    ##dataframePrettyPrint(df)

    return df

# -------------------- combine files in directory to dataframe --------------------

def combineExportFileInFolder(directory, fileName, functionName, functionArg):
    files = list(Path(directory).glob(fileName))
    if (len(files)):
        try:
            files.sort()
            if trace: print('')
            fileCount = 0
            for file in files:
                fileCount = fileCount + 1
                if trace:
                    if functionArg:
                        print(fileCount, file, '#', functionArg, '#')
                    else:
                        print(fileCount, file)
                if fileCount == 1:
                    dfc = functionName(file, functionArg)
                else:
                    dff = functionName(file, functionArg)
                    dft = pd.concat([dfc, dff], axis=0)
                    dfc = dft
        except ValueError as e:
            print('*****ValueError in file error - aborting !!')
            print (e.args)
            quit()
        except IndexError as e:
            print('*****IndexError in file error - aborting !!')
            print (e.args)
            quit()
        except TypeError as e:
            print('*****TypeError in file error - aborting !!')
            print (e.args)
            quit()
        except Exception as e:
            print('*****file error - aborting !!')
            print (e.message, e.args)
            quit()

    else:
        print('*****no %s file in folder %s - aborting !!' % (fileName, directory))
        quit()

    # sort on date, removing duplicate entries for a given day
    dfc.sort_values(['Date'], inplace=True)
    dfc = dfc.drop_duplicates(subset=['Date'])

    if trace:
        dataframePrettyPrint(dfc)

    return fileCount, dfc

# -------------------- show some table content stats --------------------

def getTableStats(conn, tableName):
    cursor = conn.execute('select count(*) from ' + tableName)
    for row in cursor:
        nbRows = row[0]
    cursor = conn.execute('select min(date), max(date) from ' + tableName)
    for row in cursor:
        dateMin = row[0]
        dateMax = row[1]
    cursor = conn.execute('select min(diff), max(diff) from ' + tableName)
    for row in cursor:
        diffMin = row[0]
        diffMax = row[1]
    print("*****table %s contains %d rows ranging %s<date<%s and %s<daily Wh<%s" %
        (tableName, nbRows, dateMin, dateMax, diffMin, diffMax))

    # total kWh for table
    cursor = conn.execute('select sum(diff) from ' + tableName)
    for row in cursor:
        diffSum = row[0]
    print("     for a current total of {:,} kWh".format(int(diffSum)))

# -------------------- store dataframe to sqlite --------------------

def storeDataframeToSql(dbName, tableName, df):
    conn = sqlite3.connect(dbName)
    df.to_sql(tableName, conn, if_exists="replace", index=False)
    getTableStats(conn, tableName)
    conn.close()
    """
    sqlite3
    .open solarPanels_data.sqlite
    .tables
    select * from sunny_boy;
    select min(date), max(date) from sunny_boy;
    select min(diff), max(diff) from sunny_boy;
    select * from sunny_boy limit 5;
    select * from enedis_conso_jour;
    select * from sqlite_master where type='table'; --> date & diff columns for the 3 tables
    .help
    .exit
    """

# -------------------- load dataframe from sqlite --------------------

def loadDataframeFromSql(dbName, tableName):
    conn = sqlite3.connect(dbName)
    getTableStats(conn, tableName)
    df = pd.read_sql('SELECT Date, Diff from ' + tableName, conn, coerce_float=True, parse_dates=('Date'))
    conn.close()
    if trace:
        dataframePrettyPrint(df)
    return df

# -------------------- plot using plotly --------------------

def plotCombined_plotly(df_purchasedEnedis, df_sendToEnedis, df_producedSolarPanels):

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_producedSolarPanels.iloc[:,0], y=df_producedSolarPanels.iloc[:,1],
        name='produced by SolarPanels',
        line=dict(color='green', width=1),
        mode='lines'))
    fig.add_trace(go.Scatter(x=df_purchasedEnedis.iloc[:,0], y=df_purchasedEnedis.iloc[:,1],
        name='purchased from Enedis',
        line=dict(color='red', width=1),
        mode='lines'))
    fig.add_trace(go.Scatter(x=df_sendToEnedis.iloc[:,0], y=df_sendToEnedis.iloc[:,1],
        name='sent to Enedis',
        line=dict(color='blue', width=1),
        mode='lines'))

    fig.update_layout(title='84 Condamines - SolarPanels vs Enedis/Linky',
        xaxis_title='Date',
        yaxis_title='kWh per day')


    fig.show()

# -------------------- main --------------------
# -------------------- combine & plot --------------------

if __name__ == '__main__':

    # just get SQL database stats
    if info:
        print("*****getting stats from database:%s" % SQLITE_DATABASE)
        conn = sqlite3.connect(SQLITE_DATABASE)
        getTableStats(conn, 'Enedis_Conso_Jour')
        getTableStats(conn, 'Enedis_Prod_Jour')
        getTableStats(conn, 'Sunny_Boy')
        conn.close()
    else:
        if source == 'xls':
            # load all csv in folder and concatenate to clean dataframes
            print("*****loading csv files in folder:%s" % CSV_XLSX_FOLDER)
            fc1_c, df1_p_c = combineExportFileInFolder(CSV_XLSX_FOLDER, 'Enedis_Conso_Jour*.csv', loadEnedisConsumedCsv, '')
            print("*****%d Enedis_Conso_Jour csv files" % fc1_c)
            fc2_c, df2_p_c = combineExportFileInFolder(CSV_XLSX_FOLDER, 'Enedis_Prod_Jour*.csv', loadEnedisProducedCsv, '')
            print("*****%d Enedis_Prod_Jour csv files" % fc2_c)
            fc3_c, df3_p = combineExportFileInFolder(CSV_XLSX_FOLDER, 'Sunny_Boy*.csv', loadSunnyBoyCsv, '')
            print("*****%d Sunny_Boy csv files" % fc3_c)
            csvFileCount = fc1_c + fc2_c + fc3_c
            print("*****%d total csv files" % csvFileCount)

            # load all xlsx in folder and concatenate to clean dataframes
            print("*****loading xlsx files in folder:%s" % CSV_XLSX_FOLDER)
            fc1_x, df1_p_x = combineExportFileInFolder(CSV_XLSX_FOLDER, '19574963805222_Export_energie_Consommation-Production_*.xlsx',
                loadEnedisXlsx, 'Export Consommation Quotidienne')
            print("*****%d Export_energie_Consommation xlsx files" % fc1_x)
            fc2_x, df2_p_x = combineExportFileInFolder(CSV_XLSX_FOLDER, '19574963805222_Export_energie_Consommation-Production_*.xlsx',
                loadEnedisXlsx, 'Export Production Quotidienne')
            print("*****%d Export_energie_Production xlsx files" % fc2_x)
            xlsxFileCount = fc1_x + fc2_x
            print("*****%d total xlsx files" % xlsxFileCount)

            # concatenate CSV and XLSX dataframes
            df1_p = pd.concat([df1_p_c, df1_p_x], axis=0)
            df2_p = pd.concat([df2_p_c, df2_p_x], axis=0)

            if False:
                df1_p_c.info(verbose=True)
                df2_p_c.info(verbose=True)
                df3_p.info(verbose=True)
                df1_p_x.info(verbose=True)
                df2_p_x.info(verbose=True)
                df1_p.info(verbose=True)
                df2_p.info(verbose=True)

            # store dataframes to sqlite
            print("*****storing dataframes to database:%s" % SQLITE_DATABASE)
            storeDataframeToSql(SQLITE_DATABASE, 'Enedis_Conso_Jour', df1_p)
            storeDataframeToSql(SQLITE_DATABASE, 'Enedis_Prod_Jour', df2_p)
            storeDataframeToSql(SQLITE_DATABASE, 'Sunny_Boy', df3_p)
        else:
            # load clean dataframes from sqlite
            print("*****loading dataframes from database:%s" % SQLITE_DATABASE)
            df1_p = loadDataframeFromSql(SQLITE_DATABASE, 'Enedis_Conso_Jour')
            df2_p = loadDataframeFromSql(SQLITE_DATABASE, 'Enedis_Prod_Jour')
            df3_p = loadDataframeFromSql(SQLITE_DATABASE, 'Sunny_Boy')

        if plot:
            print("*****plotting - close plot to terminate properly")
            plotCombined_plotly(df1_p, df2_p, df3_p)
            ##plotCombined_matplotlib(df1_p, df2_p, df3_p)
