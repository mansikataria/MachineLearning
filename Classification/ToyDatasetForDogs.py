import numpy as np
import matplotlib.pyplot as plt

#creating a toy dataset for two types of dogs,
# 500 data records for each type
greyhounds = 500
labs = 500

#average height of grehounds is 28 inches
# -- giving it some randomeness of +-4inches
grey_height = 28 + 4 * np.random.randn(greyhounds)

#average height of labradors is 24 inches
# -- giving it some randomeness of +-4inches
lab_height = 24 + 4 * np.random.randn(labs)

#visualize this in a histogram
plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()

