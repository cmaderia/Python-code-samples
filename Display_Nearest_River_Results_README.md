# Display Nearest River Results

### Usage: 
Display results for Get Distance to Nearest River (NEAR_FID and NEAR_DIST) for building centroids in a state, list of states,
or each state in a FEMA region

Results include one bar chart per state showing the binned distance to nearest river (by ranges of miles or kilometers) and the number of structure points falling into each bin. Also included is a table showing basic statistics for that state (min, max, median, mean distance to nearest river).
Both the chart and the table are exported as a single PDF for each state using the PdfPages package.

### Python version: 
3.6

### Required packages:
- [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)
- [matplotlib](https://matplotlib.org/users/installing.html)
- [numpy](https://numpy.org/)
- [os](https://docs.python.org/3/library/os.html)
- [sys](https://docs.python.org/3/library/sys.html)
- [re](https://docs.python.org/3/library/re.html)
- [glob](https://docs.python.org/3/library/glob.html)
- [datetime](https://docs.python.org/3/library/datetime.html)

### Sources:
- Source for split_uppercase [here](http://code.activestate.com/recipes/576984-split-a-string-on-capitalized-uppercase-char-using/)
- Source for createStatsTable [here](https://matplotlib.org/3.1.1/gallery/recipes/placing_text_boxes.html)
- Source for setLabels [here](https://stackoverflow.com/questions/43436595/python-reverse-dictionary?noredirect=1&lq=1)
