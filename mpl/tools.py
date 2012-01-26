import numpy as np
import matplotlib.pyplot as plt


__all__ = ['MeasureLengthTool', 'RectangularSelection']


class MeasureLengthTool(object):
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

        self.message = self._message if message == 'default' else message

        fig = ax.figure
        connect = fig.canvas.mpl_connect
        self.cids = []
        self.cids.append(connect('button_press_event', self.onpress))
        self.cids.append(connect('button_release_event', self.onrelease))
        self.cids.append(connect('motion_notify_event', self.onmove))
        self.fig = fig
        self.ax = ax
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

    def disconnect(self):
        self.ax.lines.remove(self.line)
        for c in self.cids:
            self.fig.canvas.mpl_disconnect(c)


class RectangularSelection(object):
    """Tool for measuring rectangular regions in a plot.

    Select a point in the axes to set one corner of a rectangle and drag to
    opposite corner; print the rectangle extents (xmin, xmax, ymin, ymax) when
    the mouse is released. The last length is stored as an attribute.  The
    extents are also stored in the attributes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes

    color : matplotlib.colors
        Edge color of rectangle.

    Attributes
    ----------
    xmin, xmax: float
        Minimum and maximum x-values of selected rectangle.

    ymin, ymax: float
        Minimum and maximum y-values of selected rectangle.
    """

    _message = ("Selected extents: xmin={xmin:.3g}, xmax={xmax:.3g},"
                                 " ymin={ymin:.3g}, ymax={ymax:.3g}")

    def __init__(self, ax, color='k', message='default'):

        self.message = self._message if message == 'default' else message

        fig = ax.figure
        connect = fig.canvas.mpl_connect
        self.cids = []
        self.cids.append(connect('button_press_event', self.onpress))
        self.cids.append(connect('button_release_event', self.onrelease))
        self.cids.append(connect('motion_notify_event', self.onmove))
        self.fig = fig
        self.ax = ax

        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0
        self.rect = plt.Rectangle((0, 0), 0, 0, ec=color, fc='none')
        self.ax.add_patch(self.rect)

        self.pressevent = None

    @property
    def extents(self):
        xmin = min(self.x0, self.x1)
        xmax = max(self.x0, self.x1)
        ymin = min(self.y0, self.y1)
        ymax = max(self.y0, self.y1)
        return xmin, xmax, ymin, ymax

    def onpress(self, event):
        if event.inaxes != self.ax:
            return
        self.x0 = event.xdata
        self.y0 = event.ydata
        self.rect.set_xy((self.x0, self.y0))
        self.pressevent = event

    def onrelease(self, event):
        if event.inaxes != self.ax:
            return
        self.pressevent = None

        if self.message is not None:
            xmin, xmax, ymin, ymax = self.extents
            print self.message.format(xmin=xmin, xmax=xmax,
                                      ymin=ymin, ymax=ymax)

    def onmove(self, event):
        if self.pressevent is None or event.inaxes!=self.pressevent.inaxes:
            return
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.fig.canvas.draw()

    def disconnect(self):
        self.ax.patches.remove(self.rect)
        for c in self.cids:
            self.fig.canvas.mpl_disconnect(c)

