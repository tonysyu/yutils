import matplotlib.patches as mp

from base import BaseTool


__all__ = ['RectangularSelection']


class RectangularSelection(BaseTool):
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

        BaseTool.__init__(self, ax)

        self.message = self._message if message == 'default' else message

        self.connect('button_press_event', self.onpress)
        self.connect('button_release_event', self.onrelease)
        self.connect('motion_notify_event', self.onmove)

        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0
        self.rect = mp.Rectangle((0, 0), 0, 0, ec=color, fc='none')
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

