################################################
# DisplayNearestRiverResults
#
# Functions to accompany getNearestRiverResults.ipynb.
#
# Author: Chris Maderia (cmaderia@dewberry.com)
#
# Copyright: Dewberry
#################################################


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import os
import re

# GLOBAL VARIABLES
# fields to include from input near distance and
# building attribute (SFHA) tables
INDIST_FIELDS = ['uniqueid', 'NEAR_DIST']
INSFHA_FIELDS = ['UNIQUEID', 'FLD_ZONE']

# ID fields
# near distance csv
DIST_ID = 'uniqueid'
DIST_FIELD = 'NEAR_DIST'
# building attribute csv (with flood zones)
BLDGATTR_ID = 'UNIQUEID'
BLDGATTR_FLDZONE = 'FLD_ZONE'

# output CSV results table specific field names
FIELDNAMES = ['STATE', 'STATE_ABB', 'STATE_FIPS',
              'MIN_DIST_2LESS', 'MIN_DIST_2MORE', 'MIN_DIST_ALL',
              'MAX_DIST_2LESS', 'MAX_DIST_2MORE', 'MAX_DIST_ALL',
              'MEDIAN_DIST_2LESS', 'MEDIAN_DIST_2MORE', 'MEDIAN_DIST_ALL',
              'MEAN_DIST_2LESS', 'MEAN_DIST_2MORE', 'MEAN_DIST_ALL',
              'COUNT_2LESS', 'COUNT_2MORE', 'COUNT_ALL',
              'PERCENT_2LESS', 'PERCENT_2MORE', 'PERCENT_ALL']

# Conversion factors
meters_in_a_mile = 1609.344


# FUNCTIONS
def split_uppercase(any_string):
    ''' Split any string on upper case letters
        any_string = any string
    '''
    return re.sub(r'([A-Z])', r' \1', any_string)


def setDtypes(inDf, fieldNames, datatype):
    ''' Set types for field names in a pandas data frame
        inDf = input data frame
        fieldNames = names of fields
        datatype = data type for those fields (i.e., object, int, float)
    '''
    for f in fieldNames:
        inDf[f] = inDf[f].astype(datatype)
    return inDf


def getStats(array_in_meters):
    '''Retrieve statistics in miles given an array of points
        with distances in meters
        array_in_meters = area of distances with units in meters
    '''
    # convert to miles
    array_in_miles = array_in_meters / meters_in_a_mile
    # nearest rivers 2 miles or less away
    array_in_miles_lessthan2 = array_in_miles[array_in_miles <= 2.0]
    # nearest rivers greater than 2 miles away
    array_in_miles_greaterthan2 = array_in_miles[array_in_miles > 2.0]
    arrays = [array_in_miles, array_in_miles_lessthan2,
              array_in_miles_greaterthan2]

    # calculate stats for each array in the list and format for text box
    mins = ['{:.2e}'.format(np.min(a)) if len(a) > 0 else '-' for a in arrays]
    maxes = [round(np.max(a), 2) if len(a) > 0 else '-' for a in arrays]
    medians = [round(np.median(a), 2) if len(a) > 0 else '-' for a in arrays]
    means = [round(np.mean(a), 2) if len(a) > 0 else '-' for a in arrays]
    counts = ['{:,.0f}'.format(len(a)) if len(a) > 0 else '-' for a in arrays]
    percent_totals = ['{0:.0%}'.format(len(a) / len(arrays[0]))
                      if len(a) > 0 else '-' for a in arrays]

    return arrays, mins, maxes, medians, means, counts, percent_totals


def setAxes(axs, pl, xmax):
    '''Set axes ranges and grid styles
        axs = matplotlib axis variable
        pl = matplotlib plot variable
        xmax = maximum limit of x axis range to display
    '''
    # auto set major and minor axes
    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    axs.xaxis.set_minor_locator(minor_locator_x)
    axs.yaxis.set_minor_locator(minor_locator_y)
    # set line styles for grids
    pl.grid(which='minor', linestyle=':')
    pl.grid(which='major', b=True, linestyle='-.')
    # limit x axis range from 0 to xmax
    pl.xlim(0, xmax)


def setLabels(axs, pl, figtoplot, sfhaTF, stabb, stateabb_dict):
    '''Create histogram plot labels, including title
        axs = matplotlib axis variable
        pl = matplotlib plot variable
        figtoplot = matplotlib figure
        sfhaTF = boolean value indicating centroids in SFHA / all points
        stabb = state abbreviation string (i.e., 'DC')
        stateabb_dict = dictionary of states:state abbreviations
    '''
    # set x-axis title
    axs.set_xlabel('Nearest River Distance (miles)', fontsize=12)
    # set y-axis title
    if sfhaTF:
        axs.set_ylabel('Building Centroid Count (within SFHA)', fontsize=12)
    else:
        axs.set_ylabel('Building Centroid Count', fontsize=12)
    # format axis labels with commas
    axs.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, q: format(int(y), ',')))

    # get preset x-axis labels
    figtoplot.canvas.draw()
    labels = [item.get_text() for item in axs.get_xticklabels()]

    # set the first label on the x-axis to be 0
    labels[0] = '0'
    # set these new labels
    axs.set_xticklabels(labels)

    # set title with spaces
    abb_state = {v: k for k, v in stateabb_dict.items()}
    plt.title(' '.join(re.findall('[A-Z][^A-Z]*', abb_state[stabb])),
              fontweight='bold', fontsize=18)


def readDistances(infolder, stateAbb, stateFips, mapUnmapped, keepfields):
    '''Read distance information from an input CSV file
    infolder = folder containing distance CSVs
    stateAbb = state abbreviation string
    stateFips = state fips code string
    mapUnmapped = boolean value indicating mapped or unmapped rivers
    keepfields = a list of fields to keep in the output
    '''
    dist_csv = '_' + stateFips + '_' + stateAbb + '_' + mapUnmapped + \
               '_DIST.csv'
    dist_df = pd.read_csv(os.path.join(infolder, dist_csv),
                          usecols=keepfields)
    # remove any duplicates
    dist_df = dist_df.drop_duplicates(subset=['uniqueid'])
    # return a pandas data frame with the distance information
    return dist_df


def getQueriedData(in_df, in_id, in_field, query_df, query_id,
                   query_field, query_list):
    '''Query data from a CSV (in this case, flood zones from
       building attributes CSV)
        in_df = input pandas data frame with attributes
        in_id = input unique ID field
        in_field = input field to keep
        query_df = pandas data frame to query
        query_id = query table unique ID
        query_field = query table column to base the query on
        query_list = list of attributes to include in the query
                     (i.e., A, AE, VE, for flood zones)
    '''
    # return only unique IDs with the attributes (specific flood zones)
    # in the list
    queried_ids = list(query_df.loc[query_df[query_field].isin(query_list),
                       query_id])
    # get attributes in the input data frame (Near distances) for
    # those unique ids only
    in_df = in_df.loc[in_df[in_id].isin(queried_ids), in_field]
    return in_df


def setTableFormatting(indata, tbl_cols):
    '''Set formatting for histogram table
        indata = a list of statistics in table format
                 (output from statsToTable)
        tbl_cols = column names for table
    '''
    # set colors for all cells
    cellcolors = np.empty_like(indata, dtype='object')
    for i, cl in enumerate(tbl_cols):
        if i == 0:
            cellcolors[:, i] = '#6969f0'
        elif i > 0:
            cellcolors[:, i] = '#cbcbf7'
    # set colors for column heading cells
    colcolors = np.empty_like(tbl_cols, dtype='object')
    for j, cl in enumerate(tbl_cols):
        colcolors[j] = '#6969f0'
    # set column widths
    colwidths = np.empty_like(tbl_cols, dtype='object')
    for l, cl in enumerate(tbl_cols):
        colwidths[l] = 0.06

    return cellcolors, colcolors, colwidths


def statsToTable(tbl_stats):
    '''Convert statistics to table
        tbl_stats = a list of statistics (output from getStats)
    '''
    arrays, mins, maxes, medians, means, counts, percent_totals = tbl_stats
    tbl_data = [
        ['min', mins[1], mins[2], mins[0]],
        ['max', maxes[1], maxes[2], maxes[0]],
        ['median', medians[1], medians[2], medians[0]],
        ['mean', means[1], means[2], means[0]],
        ['count', counts[1], counts[2], counts[0]],
        ['% total', percent_totals[1], percent_totals[2], percent_totals[0]]]

    return tbl_data


def createStatsTable(axs, statsToDisplay, bestFit):
    '''Create text box with stats as part of histogram plot
        axs = matplotlib axis variable
        statsToDisplay = a list of statistics (output from getStats)
        bestFit = a boolean value indicating whether to use best fit
                  for the histogram
    '''
    table_data = statsToTable(statsToDisplay)
    # set column names
    columns = ['', '<=2 mi', '> 2 mi', 'All']
    # adjust table location so it overlaps less with the bins
    means = table_data[3]
    if bestFit:
        # if the means of dist >= 2 is greater than 1
        if means[1] > 1.0:
            table_loc = 'upper left'
        else:
            table_loc = 'upper right'
    elif bestFit is False:
        table_loc = 'upper right'

    cell_colors, col_colors, col_widths = \
        setTableFormatting(table_data, columns)
    table = axs.table(cellText=table_data, colLabels=columns,
                      cellColours=cell_colors, colColours=col_colors,
                      colWidths=col_widths, loc=table_loc,
                      zorder=3, cellLoc='left')
    # font size and scaling
    table.set_fontsize(9)
    table.scale(1, 1)
    # other table formatting
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.25)
        cell.set_edgecolor('#6969f0')


def createHistogram(inStats, sfhaTF, stabb, stabb_dict, pdfpp,
                    bestfit, numBins=100):
    '''Create histogram - user can specify number of bins and all points
    or those just inside of SFHA, everything else on plot is preset
    inStats = a list of statistics (output from getStats)
    sfhaTF = boolean value indicating centroids in SFHA / all points
    stabb = state abbreviation string (i.e., 'DC')
    stabb_dict = dictionary of state:state abbrevations
    pdfpp = variable used by PdfPages to store output that it will save
            to a pdf file
    bestfit = a boolean value indicating whether to use best fit for
              the histogram
    numBins = number of bins to show on histogram plot
    '''
    # read in the stats - lists (all, 2 miles or less, greater than 2 miles)
    arrays, mins, maxes, medians, means, counts, percent_totals = inStats
    # set up the plot
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    x, bins, p = ax.hist(arrays[1], bins=numBins, align='mid', color='blue')

    # create axes, labels, and stats table
    setAxes(ax, plt, 2)
    setLabels(ax, plt, fig, sfhaTF, stabb, stabb_dict)
    createStatsTable(ax, inStats, bestfit)

    fig.tight_layout()
    pdfpp.savefig(bbox_inches='tight')
    plt.close(fig)


def initializeTable(colnames):
    '''Create a new pandas data frame from scratch
    colnames = names of columns
    '''
    outdf = pd.DataFrame(index=None, columns=colnames)
    setDtypes(outdf, colnames, 'object')
    return outdf


def addTableRow(indf, colnames, inStats, statename, stateabb, statefips):
    '''Add a row to a pandas data frame
    indf = input data frame
    colnames = names of columns
    inStats = a list of statistics (output from getStats)
    statename = state name string (i.e., 'District Of Columbia')
    stateabb = state abbreviation string (i.e., 'DC')
    statefips = state fips string (i.e., '11')
    '''
    # read in the stats - lists (all, 2 miles or less, greater than 2 miles)
    arrays, mins, maxes, medians, means, counts, percent_totals = inStats
    table_data = statsToTable(inStats)

    # write data to the row
    rowToWrite = [split_uppercase(statename).lstrip(), stateabb, statefips]
    for i in list(range(0, 6)):
        rowToWrite = rowToWrite + table_data[i][1:]
    nextdf = pd.DataFrame(data=[rowToWrite], columns=colnames)

    return nextdf
