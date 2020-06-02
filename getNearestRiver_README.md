# Get Distance to Nearest Rivers

### Usage:
Get Distance to Nearest River (NEAR_FID, NEAR_DIST, NEAR_X, NEAR_Y) for building centroids in a state or list of states in a FEMA region
C:\Python27\ArcGISx6410.5> python <file_directory>\getNearestRiver_20190823.py    (make sure QuickEdit mode is turned off before running)

### Python version: 
2.7

### Required packages:
- [arcpy](https://desktop.arcgis.com/en/arcmap/latest/analyze/python/importing-arcpy.htm)
- [functools](https://docs.python.org/3/library/functools.html)
- [glob](https://docs.python.org/3/library/glob.html)
- [multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
- [os](https://docs.python.org/3/library/os.html)
- [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)
- [geopandas](http://geopandas.org/install.html)
- [time](https://docs.python.org/3/library/time.html)
- [datetime](https://docs.python.org/3/library/datetime.html)

### Notes:
Designed to run as a stand-alone script in the Command Prompt (but can be run in Jupyter Lab with minor modification - creation of a companion Python script, arcpy_workers.py, with the getNearFeatures function, which is input into the multiprocessing.Pool)


## Sources:
- Source for filter_fields [here](https://community.esri.com/thread/56589)
- Source for formatting the where clause in getRiversSubset [here](https://community.esri.com/thread/205245-arcpyselectlayerbyattributes-where-clause)
- Source for getChunks [here](https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks)
- Source for JoinField [here](https://gis.stackexchange.com/questions/95957/most-efficient-method-to-join-multiple-fields-in-arcgis)
