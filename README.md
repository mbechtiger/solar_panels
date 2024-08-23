## Solar panels - plot consumption vs production

I use Python+Pandas to import CSV files produced by the french power provider Enedis (consumption & production), my Sunny Boy solar panels inverter. Then Sqlite to store the cleaned up data for later use. And finally Plotly to create a graph of the merged data. 

Note that the CSV are extracted from the sources given a date range and merged - I treat mote than 40 CSV files exported since 2021. I initially used Matplotlib for plotting but didn't manage to fix an unexplained display bug (at the end of the date range).

All CSV files are exported on-demand, I couldn't automate the data exports using some API.

- "plot_purchased_produced.py" merges/imports all CSV in the hard-coded folder, stores them in Sqlite, plots using Plotly. Check the "--help" for parameters
- "solarPanels_data.sqlite" contains the current data, which can be reused for plotting invoking the "--source SQL" parameter
- "Sunny_Boy_xxx.csv" is a sample inverter export file
- "Enedis_Prod_Jour_xxx.csv" is a sample production (as received on the electrical supplier's network) export file
- "Enedis_Conso_Jour_xxx.csv" is a sample consumption (as produced on the electrical supplier's network) export file
- "Screen Shot 2024-08-21 at 20.50.30.jpg" shows a sample Plotly graph

Contact me at marcelbechtiger@gmail.com.
