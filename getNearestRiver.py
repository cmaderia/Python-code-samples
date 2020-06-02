##############################################################################
# Find Distance to Nearest Rivers
#
# Usage: Find location and distance to Nearest River
# for building centroids in a state, list of states,
#
# Author: Chris Maderia (cmaderia@dewberry.com)
#
# Copyright: Dewberry
###############################################################################

# Import packages
import arcpy
import os
import numpy as np
from glob import glob
import multiprocessing as mp
from multiprocessing import Pool
import logging
import pandas as pd
import geopandas as gpd
import shutil
import sys
from time import time
from datetime import date

sys.path.append('../../')
from core.ConventionOps import state_codes, stateabb, \
    regstates, state_adjState

# Start timer
start = time()
# Check out the ArcGIS Spatial Analyst extension license
arcpy.CheckOutExtension('Spatial')

# list of states or use regstates dict to do FEMA regions (1-10)
processingStateList = regstates['1'] + regstates['2']

# list of states to exclude
states_to_remove = []

inputs_Folder = r'C:\temp\CNMS'
riversGdb = r'CNMS_All_Regions_FY19Q4.gdb\CNMS_Inventory_Albers'
centroidsGdb = r'Bldg_ftprnt_centroids.gdb'
noCentroids_csv_Folder = r'T:\CCSI\TECH\FEMA\FEMA_HQ_Queries' + \
    r'\FEMA_Mission_Support\Tier0_analytics\Pop_allocation\Justin'

# building attribute / flood zone inputs
bldg_attr_Root = r'T:\CCSI\TECH\FEMA\FEMA_HQ_Queries\FEMA_Mission_Support\
    Tier0_analytics\Building_attributes\10112019'
# location to save files -  output directory will be created here
outRoot = r'C:\temp\CNMS'
# name of output directory
outFiles = 'cnms_near_rivers_' + ''.join(str(date.today()).split('-'))
# True = Use arcpy.Near_analysis, False = Use arcpy.GenerateNearTable
nearAnalysis = True
# True = S_Studies_Ln (mapped), False = S_Unmapped_Ln (unmapped)
studiedLines = True
# True = run for bldgs with parcels only (no centroids/uniqueid),
# False = run for bldgs with centroids/uniqueid (standard)
noCentroids = False

# fields to include
keepFields_Bldgs = ['OBJECTID', 'uniqueid', 'PARCEL_ID']
if studiedLines:
    keepFields_river = ['REACH_ID', 'WTR_NM', 'FLD_ZONE', 'VALIDATION_STATUS',
                        'STATUS_TYP', 'MILES', 'ST_FIPS', 'CO_FIPS']
else:
    keepFields_river = ['UML_ID', 'GNIS_NAME', 'MILES', 'CO_FIPS', 'ST_FIPS']

# remove states not needed
for st in states_to_remove:
    if st in processingStateList:
        processingStateList.remove(st)

# River Lines - these should already be projected into the
    # coordinate system of choice,
    # Albers is preferred
riversRoot = os.path.join(inputs_Folder, riversGdb)
arcpy.env.workspace = riversRoot
if studiedLines:
    riverLines = os.path.join(riversRoot, str(
        arcpy.ListFeatureClasses('S_Studies_Ln*')[0]))
else:
    riverLines = os.path.join(riversRoot, str(
        arcpy.ListFeatureClasses('S_Unmapped_Ln*')[0]))

# Building Centroids from Bing
centroidsRoot = os.path.join(inputs_Folder, centroidsGdb)

# Get number of cores
# total number of CPU threads on the machine - 3 is optimal
# (or 1/4 the number of cores, but greater than 2)
# observations show that this provides optimal performance
cpu_cnt = [mp.cpu_count()]
num_cores = 3
# int([3 if round(i / 4) < 3 else round(i / 4) for i in cpu_cnt][0])

# Conversion factors
meters_in_a_mile = 1609.344


# Functions

def glob_recursive(glob_folder, filter_strings):
    '''Return a list of all files in a folder, using a filter
    glob_folder = folder to search
    filter_strings = a list of strings to filter on
    '''
    filelist = []
    for root, dirs, files in os.walk(glob_folder, topdown=False):
        for name in files:
            if all(fs in name for fs in filter_strings):
                filelist.append(os.path.join(root, name))
    return filelist


def drop_dups(inFeatures, dupField, no_centroids,
              outFolder='', outBaseName='', outType=''):
    """
    Drops duplicates from a feature class (in a File GDB), a shapefile,
    or a table (in a File GDB), with an optional output to a CSV or
    a shapefile.
    If no output is specified, then a pandas data frame is returned
    Note: This uses geopandas (see gist for installation),
    Field names may be shortened in shapefile output.

    inFeatures = input shapefile, feature class, or File GDB table
    dupField = field to search for duplicates
    outFolder = output folder (optional)
    outBaseName = output CSV/shapefile name (optional)
    outType = output file type - 'csv' or 'shapefile' (optional)

    returns a pandas Dataframe
    """
    # Generate a pandas data frame of the input features (no geometry)
    gdbname = os.path.split(inFeatures)[0]
    basename = os.path.split(inFeatures)[1]
    try:
        out_gdf = gpd.read_file(gdbname, driver='FileGDB', layer=basename,
                                encoding='utf-8')
    except MemoryError:
        out_gdf = gpd.read_file(gdbname, driver='FileGDB', layer=basename)

    if not no_centroids:
        # Remove any duplicates using pandas
        print('With duplicates: ', len(out_gdf))
        out_gdf = out_gdf.drop_duplicates(subset=[dupField])
        print('Without duplicates: ', len(out_gdf))

    # output to CSV or shapefile
    if outBaseName == '':
        outBaseName = basename
    if outType.lower() == 'csv':
        outTableCsv = os.path.join(outFolder, outBaseName + '.csv')
        out_df = pd.DataFrame(out_gdf)
        out_df.to_csv(outTableCsv, encoding='utf-8')
        del out_df
    elif outType.lower() == 'shapefile':
        outShp = os.path.join(outFolder, outBaseName + '.shp')
        out_gdf.to_file(outShp)
        del out_df
    else:
        pass
    return pd.DataFrame(out_gdf)


def filter_fields(FC, fieldList):
    '''Filter fields in a feature class (include only those in fieldList)
    FC = the feature class
    fieldList = the list of fields to keep
    '''
    # list of fields in feature class
    fields = arcpy.ListFields(FC)
    # arcpy FieldInfo object
    fieldinfo = arcpy.FieldInfo()
    for field in fields:
        if field.name not in fieldList:
            # hide fields that are not in fieldList
            fieldinfo.addField(field.name, field.name, 'HIDDEN', '')
    return fieldinfo


def makeCentroidsLyr(incsv, xfield, yfield, outlayer):
    '''Create a geospatial layer from a CSV with X, Y (lat, lon)
    coordinates, using ArcPy
    incsv = the input CSV file with x, y coordinates
    xfield = the latitude (x) field
    yfield = the longitude (y) field
    outlayer = the output layer file (in memory)
    '''

    outlayer = os.path.join('in_memory', outlayer)

    # read in the CSV
    incsv_df = pd.read_csv(incsv)
    # keep only needed fields
    incsv_df = incsv_df[keepFields_Bldgs[1:3] + [xfield, yfield]]

    # limit to only rows with no uniqueid
    incsv_df = incsv_df[incsv_df['uniqueid'].isnull()]

    # create an OBJECTID field that starts at 1 and
    # goes to max record count, just like in ArcGIS
    incsv_df['OBJECTID'] = range(1, 1 + len(incsv_df))

    x = np.array(np.rec.fromrecords(incsv_df.values))
    names = incsv_df.dtypes.index.tolist()
    x.dtype.names = tuple(names)
    arcpy.da.NumPyArrayToTable(x, outlayer)

    arcpy.MakeXYEventLayer_management(outlayer, xfield,
                                      yfield, outlayer)
    return outlayer


def createLayer(inFC, lyrcount, savefolder, fields_to_keep, defquery=''):
    '''Copy features to a layer in memory
        inFC = input feature class or shapefile
        lyrcount = number of layer (ex., 1 if just one layer;
            if multiple then use count+=1 in a loop);
            used to prevent previous layer from being overwritten when looping
        savefolder = folder where layer or feature class will be saved
        fields_to_keep = which fields to keep
        defquery = a SQL where statement that can be applied to limit
        the number of output features
    '''
    # name of layer to create
    lyrname = str(os.path.basename(inFC)) + '_' + str(lyrcount)
    # location to save layer locally
    layer_local = os.path.join(savefolder, lyrname)
    if arcpy.Exists(lyrname):
        arcpy.Delete_management(lyrname)
    # filter fields and make layer
    if len(fields_to_keep) > 0:
        fieldInfo = filter_fields(inFC, fields_to_keep)
        arcpy.MakeFeatureLayer_management(in_features=inFC, out_layer=lyrname,
                                          field_info=fieldInfo)
    else:
        arcpy.MakeFeatureLayer_management(in_features=inFC, out_layer=lyrname)
    lyr = arcpy.mapping.Layer(lyrname)
    lyr.name = lyrname
    lyr.definitionQuery = defquery

    # save rivers as layer file
    if 's_studies_ln' in lyrname.lower() or 's_unmapped_ln' in lyrname.lower():
        lyr.saveACopy(os.path.join(savefolder, layer_local))
        fileExt = '.lyr'
    # otherwise, save points as feature classes in individual
        # GDBs (to avoid locks)
    # default coordinate system = Albers Equal Conic Area
    else:
        arcpy.env.workspace = savefolder
        outlyrGdb = setup_folder(savefolder, lyrname + '.gdb', 'gdb')
        arcpy.env.workspace = outlyrGdb
        arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(
            'North America Albers Equal Area Conic')
        layer_local = os.path.join(outlyrGdb, lyrname)
        if arcpy.Exists(layer_local):
            arcpy.Delete_management(layer_local)
        arcpy.CopyFeatures_management(lyr, layer_local)
        # file extension - blank for feature class, '.shp' for shapefile
        fileExt = ''
    # return location of layer or feature class
    return layer_local + fileExt


def getRiversSubset(statename, st_abb_dict, st_fips_dict, adjstates,
                    riversFC, keepfields, folderout):
    '''create a subset of river features based on state FIPS
        code for defined input state
        and neighboring states (used in conjunction with the
        createLayer function)
        statename = name of input state
        st_abb_dict = dictionary of state:abbreviation
        st_fips_dict = dictionary of abbrevation:fips code
        adjstates = dictionary of state:adjacent states
        riversFC = input rivers features
        keepfields = fields to keep (others will be hidden and not exported)
        folderout = location to save the output layer
    '''
    st_abb = st_abb_dict[statename]
    # current state abbreviation + adjacent state abbreviations in a list
    adj_st_abb_list = [st_abb] + list(adjstates[st_abb])
    adjstates_fips_list = [st_fips_dict[a] for a in adj_st_abb_list]
    # format list for SQL where clause in arcpy
    adjstates_fips_list_arcpy = \
        ",".join(["'{}'".format(k) for k in adjstates_fips_list])
    # where clause used for definition query or selection of features
    where = arcpy.AddFieldDelimiters(riversFC, "ST_FIPS") + \
        """ IN {0}""".format("(" + adjstates_fips_list_arcpy + ")")
    # create rivers layer
    riversLyr = createLayer(riversFC, 1, folderout, keepfields, where)
    # return location of output rivers layer
    return riversLyr


def getChunks(r, n):
    '''Yield successive n-sized chunks for a certain range r
        r = a range of indices (i.e., [1,2,3,4,... 5000])
        n = size of each chunk (i.e., 500)
    '''
    # use range instead of xrange in Python 3
    for i in xrange(0, len(r), n):
        yield r[i:i + n]


def loadCentroids(centroidsFC, keepfields, outfolder, no_centroids, cores):
    '''loads a centroids layer and saves the output in [number of cores] chunks
        centroidsFC = input centroids feature class
        keepfields = fields to keep in the output
        outfolder = where to save the output
        no_centroids = boolean variable that indicates
            if parcel-derived centroids
                       (no uniqueid) will be used (yes=True)
        cores = number of cores; used to determine number of
            chunks and chunk size
    '''
    # make a copy of the centroids features in memory
    centroidLyr = str(os.path.basename(centroidsFC)) + '_Layer'
    arcpy.MakeFeatureLayer_management(centroidsFC, centroidLyr)
    if no_centroids:
        oidField = 'OBJECTID'
    else:
        oidField = arcpy.Describe(centroidLyr).OIDFieldName
    # get the count of features
    centroid_count = int(arcpy.GetCount_management(centroidLyr)[0])
    print(centroid_count)

    centroidsLyr_list = []
    if centroid_count > 0:
        chunksize = (centroid_count) / cores
        # get list of lists (one list of consecutive counts per core) -
        # these should match the OBJECTIDs
        chunkslist = list(getChunks(range(1, centroid_count + 1), chunksize))

        # if there is an extra list of OBJECTIDs (i.e., a 13th list and
        # there is only 12 cores),
        # then append that last list to the one before it
        if (len(chunkslist) > cores):
            if len(chunkslist[cores]) > 0:
                chunkslist[cores - 1] = \
                    chunkslist[cores - 1] + chunkslist[cores]
            del chunkslist[cores]

        for count, chunklist in enumerate(chunkslist):
            print '(' + str(count + 1) + '/' + str(cores) + ')...',
            # where clause to query and get chunks
            where = arcpy.AddFieldDelimiters(centroidLyr, oidField) + \
                """ > {0} AND {1} < {2}""".format(min(chunklist) - 1, oidField,
                                                  max(chunklist) + 1)
            # create chunk by applying where clause to centroids layer
            centroidsLyr = createLayer(centroidLyr, count + 1, outfolder,
                                       keepfields, where)
            centroidsLyr_list.append(centroidsLyr)
    else:
        print('No centroid features')
    arcpy.Delete_management(centroidLyr)
    # return a list of paths to centroid layers that are saved locally
    return centroidsLyr_list


def setup_folder(mainDir, foldername, folderOrGdb):
    '''Create a folder or file geodatabase (GDB) for processing
        maindDir = directory in which to store the folder or GDB
        foldername = name of folder or GDB to create
        folderOrGdb = specify whether to create a folder or a GDB
    '''
    if folderOrGdb == 'folder':
        folderPath = os.path.join(mainDir, foldername)
        if not os.path.exists(folderPath):
            os.mkdir(folderPath)
    elif folderOrGdb == 'gdb':
        folderPath = os.path.join(mainDir, os.path.basename(foldername))
        if not arcpy.Exists(folderPath):
            arcpy.CreateFileGDB_management(
                mainDir, os.path.basename(foldername))
    # return the path to the folder
    return folderPath


def JoinField(inTable, inJoinField, joinTable, outJoinField, joinFields):
    '''Join fields from an outside table or feature class to
        the input table or feature class
        inTable = input table or feature class to join the fields to
        inJoinField = input table or feature class field to base the join on
        joinTable = outside table or feature class to get
        the fields to join from
        outJoinField = outside table or feature class field to
        base the join on
        joinFields = list of fields to join from the outside
        table or feature class
    '''
    # Step 1: Read the join table creating a dictionary with
    # vnames of the join fields as keys
    # and the values in each field as dictionary values
    joinfields = [outJoinField] + joinFields.split(';')
    joindict = {}
    with arcpy.da.SearchCursor(joinTable, joinfields) as rows:
        for row in rows:
            joinval = row[0]
            joindict[joinval] = [row[i] for i in range(len(row))[1:]]
    del row, rows

    # Step 2: Create new fields in the target shapefile
    fList = [f for f in arcpy.ListFields(joinTable)
             if f.name in joinFields.split(';')]
    for i in range(len(fList)):
        name = fList[i].name
        type = fList[i].type
        if type in ['Integer', 'OID']:
            arcpy.AddField_management(inTable, name, field_type='LONG')
        elif type == 'String':
            arcpy.AddField_management(
                inTable, name, field_type='TEXT', field_length=fList[i].length)
        elif type == 'Double':
            arcpy.AddField_management(inTable, name, field_type='DOUBLE')
        elif type == 'Date':
            arcpy.AddField_management(inTable, name, field_type='DATE')

    # Step 3: Update the newly added fields in the target table or
    # feature class with
    # the values from those same fields in the other table or feature class
    targfields = [inJoinField] + joinFields.split(';')
    with arcpy.da.UpdateCursor(inTable, targfields) as recs:
        for rec in recs:
            keyval = rec[0]
            # Step 4: Use the if/else statements to join and
            # populate the target fields
            if keyval in joindict:
                for i in range(1, len(targfields)):
                    rec[i] = joindict[keyval][i - 1]
                rec = tuple(rec)
            recs.updateRow(rec)
    del recs, rec


def getNearFeatures(arg):
    '''Get Nearest River features for building centroids
        (used as input to the multiprocessing pool)
        arg = a series of arguments to feed into the multiprocessing pool
        *Important: arg should be in the following format:
        ((c, riverLines, outfolder) for c in inPoints)
        inPoints = the list of file locations for each centroid chunk
        riverLines = location of the newly created subset of river
        line features
        outfolder = where to store the output table or feature
        class with added Near fields
    '''
    inPoints, riverLines, outfolder = arg

    # set up outputs for GenerateNearTable_analysis function
    outChunkGdb = setup_folder(
        outfolder, str(inPoints).replace('Layer', 'Table') + '.gdb', 'gdb')
    arcpy.env.workspace = outChunkGdb
    arcpy.env.outputCoordinateSystem = \
        arcpy.SpatialReference('North America Albers Equal Area Conic')
    outTableName = os.path.basename(
        inPoints).split('.')[0].replace('Layer', 'Table')
    outTable = os.path.join(outChunkGdb, outTableName)

    # if user specifies run Near_analysis tool, then run (feature class output)
    if nearAnalysis:
        arcpy.Near_analysis(
            inPoints, riverLines, location='LOCATION', method='GEODESIC')
    # if user specifies not to run Near_analysis tool,
    # then run GenerateNearTable (table output)
    elif not nearAnalysis:
        arcpy.GenerateNearTable_analysis(
            inPoints, riverLines, outTable,
            location='location', method='GEODESIC')


if __name__ == '__main__':

    # Create the folder where the output files will go
    mainfolder = setup_folder(outRoot, outFiles, 'folder')
    # Create the Table folder where the output CSVs will go
    outTableFolder = setup_folder(
        mainfolder, 'Tables_' + os.path.basename(
            riverLines).split('_')[1], 'folder')

    # Start logger
    logOut = os.path.join(mainfolder, 'cnms_River_log.txt')
    logging.basicConfig(filename=logOut, level=logging.INFO)

    for state in processingStateList:
        # Start timer for state
        startState = time()
        # Remove white space from state name up front
        # (formatting to match input filenames and dictionaries)
        state = ''.join(state.split())
        logging.info(state)
        print(state)

        # Create the state processing folder and subfolders
        # (where intermediate outputs will be stored)
        statefolder = setup_folder(mainfolder, state, 'folder')
        pointsChunkDir = setup_folder(statefolder, 'chunks', 'folder')
        tableChunkDir = setup_folder(statefolder, 'table_chunks', 'folder')

        # Identifier for Unmapped or Studied river lines
        unmappedOrStudied = os.path.basename(riverLines).split('_')[1]
        # Location of building centroids for that state
        arcpy.env.workspace = centroidsRoot

        # Get centroids for the state
        # for only parcel-derived centroids
        if noCentroids:
            csvlist = glob_recursive(
                noCentroids_csv_Folder, [stateabb[state], '.csv'])
            noCentroids_csv = csvlist[0]
            state_centroids = makeCentroidsLyr(
                noCentroids_csv, 'X', 'Y', stateabb[state] + '_noCentroids')
        # for all other centroids
        else:
            state_centroids = os.path.join(centroidsRoot, str(
                arcpy.ListFeatureClasses(
                    stateabb[state] + '*')[0]))

        # Create feature class chunks of the centroids, and compile
        # paths of chunks into a list
        centroid_lyrs_list = loadCentroids(state_centroids, keepFields_Bldgs,
                                           pointsChunkDir, noCentroids,
                                           num_cores)

        # If there are centroid chunks
        if len(centroid_lyrs_list) > 0:

            # Set output location for merged table, prior to
            # joining River lines fields
            outTableGdb = setup_folder(statefolder, state + '_nearestRiver_' +
                                       unmappedOrStudied + '.gdb', 'gdb')
            outTable = os.path.join(outTableGdb, state + '_nearestRiver_' +
                                    unmappedOrStudied)

            # Create rivers layer for that state
            rivers_lyr = getRiversSubset(state, stateabb, state_codes,
                                         state_adjState, riverLines,
                                         keepFields_river, pointsChunkDir)

            # Make the Pool of workers
            logging.info('\nNear...')
            print('\nNear...'),

            # create arguments to input into the pool
            args = ((c, rivers_lyr, tableChunkDir) for c in centroid_lyrs_list)
            pool = Pool(processes=num_cores)
            # assign each chunk to a worker in the pool
            pool.map_async(getNearFeatures, args)

            # close the pool and wait for the work to finish
            pool.close()
            pool.join()

            # output points with Near information attached
            outPoints = [os.path.join(p, os.path.basename(p).split('.')[0])
                         for p in glob(os.path.join(pointsChunkDir, '*.gdb'))]

            # If using Generate Near Table function, use output
            # tables with Near info
            if not nearAnalysis:
                # output tables with Near information attached
                outTables = [os.path.join(t, os.path.basename(t).split('.')[0])
                             for t in glob(os.path.join(tableChunkDir,
                                           '*.gdb'))]

                logging.info('Joining layer chunk fields to table chunks...')
                print('Joining fields...'),

                # remove OBJECTID field from building centroids fields list
                keepFields_Bldgs.remove('OBJECTID')
                # loop through output tables and join ID
                # fields from output points
                for op in outPoints:
                    ot = op.replace('Layer', 'Table').replace(
                        'chunks', 'table_chunks')
                    pointsJoinOn = arcpy.Describe(op).OIDFieldName
                    tableJoinOn = 'IN_FID'
                    fieldsToJoin = ';'.join(keepFields_Bldgs)
                    JoinField(ot, tableJoinOn, op, pointsJoinOn, fieldsToJoin)
            # otherwise use output points with Near info
            else:
                outTables = outPoints

            logging.info('Merging outputs...')
            print('Merging...'),
            arcpy.env.workspace = outTableGdb
            arcpy.Merge_management(outTables, outTable)

            print('Joining river fields...')
            tablesJoinOn = 'NEAR_FID'
            riversJoinOn = arcpy.Describe(riverLines).OIDFieldName
            riversJoinFields = ';'.join(keepFields_river)
            JoinField(outTable, tablesJoinOn, riverLines,
                      riversJoinOn, riversJoinFields)

            # Remove duplicates
            out_df = drop_dups(os.path.join(outTableGdb, outTable),
                               'uniqueid', noCentroids)

            # Add Near Distance in Miles field
            out_df['NEAR_DIST_MILES'] = out_df['NEAR_DIST'] / meters_in_a_mile

            # output to CSV
            if unmappedOrStudied == 'Studies':
                unmappedOrStudied = 'Mapped'
            if noCentroids:
                outTableCsv = os.path.join(outTableFolder, state +
                                           '_nearestRiver_' +
                                           unmappedOrStudied + '_noUID.csv')
            else:
                outTableCsv = os.path.join(outTableFolder, state +
                                           '_nearestRiver_' +
                                           unmappedOrStudied + '.csv')
            # drop extra columns
            for c in ['Unnamed: 0', 'geometry']:
                if c in out_df.columns:
                    out_df = out_df.drop([c], axis=1)
            out_df.to_csv(outTableCsv, encoding='utf-8')
            del out_df

        # Otherwise, if no centroid chunks, then skip this state
        else:
            print('Skipping ' + state + '.' + ' \
                No building centroid chunks found!')

        # remove the state folder with intermediate files/folders
        logging.info('Deleting the intermediate files/folders...')
        if os.path.exists(statefolder):
            shutil.rmtree(statefolder, ignore_errors=True)

        # end timer
        logging.info(str((time() - startState) / 60) + ' minutes')
        print(str((time() - startState) / 60) + ' minutes')

    # delete extra files in output folder that are not CSVs
    os.chdir(outTableFolder)
    for removeFile in glob('*'):
        if '.csv' not in removeFile:
            try:
                os.remove(removeFile)
            except OSError:
                pass

    logging.info(str((time() - start) / 60) + ' minutes TOTAL')
    print(str((time() - start) / 60) + ' minutes TOTAL')

    logging.info('Your output location is here: \n' + outTableFolder)
    print('Your output location is here: \n' + outTableFolder)
