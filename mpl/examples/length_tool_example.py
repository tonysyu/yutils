import numpy as np
import matplotlib.pyplot as plt

from yutils.mpl.tools import MeasureLengthTool


plt.ion()
f, ax = plt.subplots()
plt.imshow(np.random.random((20, 20)), interpolation='nearest')

rect = MeasureLengthTool(ax)
plt.draw()
raw_input('enter to continue\n')

