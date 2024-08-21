## Solar panels - plot consumption vs production

I use Python+Pandas to import CSV files produced by the french power provider Enedis (consumption & production), my Sunny Boy solar panels inverter. Then Sqlite to store the cleaned up data for later use. And finally Plotly to create a graph on the merged data. Note that the CSV are extracted from the sources given a date range - I treat mote than 40 CSV files created since 2021. I started ploting with Matplotlib but didn't manage to fix an unexplained display bug.

All CSV files are exported on-demand, I didn't automate the data retrieval using some API.

- "plot_purchased_produced.py" merges/imports all CSV in the hard-coded folder, stores them in Sqlite, plots using Plotly. Check the "--help" for parameters
- "solarPanels_data.sqlite" contains the current data and can be used invoking the "--source SQL" parameter
- "Sunny_Boy_xxx.csv" is a sample inverter export file
- "Enedis_Prod_Jour_xxx.csv" is a sample production (as received on the electrical supplier's network) export file
- "Enedis_Conso_Jour_xxx.csv" is a sample consumption (as produced on the electrical supplier's network) export file

Contact me at marcelbechtiger@gmail.com.
