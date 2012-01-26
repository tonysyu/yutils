import numpy as np

from base import BaseTool


__all__ = ['MeasureLengthTool']


class MeasureLengthTool(BaseTool):
    """Tool for measuring lengths in a plot.

    Select a point in the axes and drag to measure a length; print the selected
    length when the mouse is released. The last length is stored as an
    attribute.

    Fixme: Adding a line to an image plot tends to pad images, which leaves
           ugly whitespace.

    Parameters
    ----------
    ax : matplotlib.axes.Axes

    color : matplotlib.colors
        Line color.

    Attributes
    ----------
    length : float
        Length of selected line.
    """

    _message = 'Selected length: %g pixels'

    def __init__(self, ax, color='k', message='default'):

        BaseTool.__init__(self, ax)

        self.message = self._message if message == 'default' else message

        self.connect('button_press_event', self.onpress)
        self.connect('button_release_event', self.onrelease)
        self.connect('motion_notify_event', self.onmove)
        self.line, = ax.plot((0, 0), (0, 0), color=color)
        self.pressevent = None

    def onpress(self, event):
        if event.inaxes != self.ax:
            return
        self.x0 = event.xdata
        self.y0 = event.ydata
        self.line.set_data([self.x0, self.x0], [self.y0, self.y0])
        self.fig.canvas.draw()
        self.pressevent = event

    def onrelease(self, event):
        if event.inaxes != self.ax:
            return
        self.pressevent = None
        x1 = event.xdata
        y1 = event.ydata
        self.length = np.sqrt((x1 - self.x0)**2 + (y1 - self.y0)**2)
        if self.message is not None:
            print self.message % self.length

    def onmove(self, event):
        if self.pressevent is None or event.inaxes!=self.pressevent.inaxes:
            return
        x1 = event.xdata
        y1 = event.ydata
        self.line.set_data([self.x0, x1], [self.y0, y1])
        self.fig.canvas.draw()

