

import numpy as np
import matplotlib.pyplot as plt


import pandas
data 	= 	pandas.read_csv("stocknews/Combined_News_DJIA.csv")



N = 2
menMeans = (data[data.Label == 1].shape[0], data[data.Label == 0].shape[0])

data[data.Label == 1].shape[0]

ind = np.arange(N)  # the x locations for the groups
       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, menMeans,align='center', alpha=0.5, color='gr')



# add some text for labels, title and axes ticks
ax.set_ylabel('# of days')
ax.set_title('Class Distribution')
ax.set_xticks(ind)
ax.set_xticklabels(('Label = 1', 'Label = 0'))

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')


plt.ylim(0,data.shape[0])
autolabel(rects1)
plt.savefig("class-distrubtuion.png")

#plt.show()