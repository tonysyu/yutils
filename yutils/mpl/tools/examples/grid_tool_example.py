import numpy as np
import matplotlib.pyplot as plt

from yutils.mpl.tools import GridTool


# NOTE: This tool behaves strangely when you click the figure before it is
# completely drawn.

plt.ion()
fig, ax = plt.subplots()
ax.imshow(np.random.random((20, 20)), interpolation='nearest')
ax.set_title("Click and drag a line to move it; "
             "'h' or 'v' to add line; "
             "'d' to delete selected line.")
gridtool = GridTool(ax)
plt.draw()
raw_input('enter to continue\n')
print '\nxdata = ', ['%.3g' % x for x in gridtool.xdata]
print '\nydata = ', ['%.3g' % y for y in gridtool.ydata]

