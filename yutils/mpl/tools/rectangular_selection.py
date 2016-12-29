import warnings
import matplotlib.patches as mp

from .base import BaseTool


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

    edgecolor : matplotlib.colors
        Edge color of rectangle.

    facecolor : matplotlib.colors
        Edge color of rectangle.

    alpha : 0 < float <= 1
        Alpha value (i.e. opacity) of rectangle.

    message : str
        Format string for printing rectangle extents. `str.format()` will be
        called on this string, with 'xmin', 'xmax', 'ymin', 'ymax' keys.
        If None, no message is printed.

    color : matplotlib.colors
        Edge color of rectangle. (Deprecated)

    Attributes
    ----------
    xmin, xmax: float
        Minimum and maximum x-values of selected rectangle.

    ymin, ymax: float
        Minimum and maximum y-values of selected rectangle.
    """

    _message = ("Selected extents: xmin={xmin:.3g}, xmax={xmax:.3g},"
                                 " ymin={ymin:.3g}, ymax={ymax:.3g}")

    def __init__(self, ax, edgecolor='k', facecolor='b', alpha=0.2,
                 message='default', color=None):

        BaseTool.__init__(self, ax)

        if color is not None:
            warnings.warn("`color` parameter deprected, use `edgecolor`")
        self.message = self._message if message == 'default' else message

        self.connect('button_press_event', self.onpress)
        self.connect('button_release_event', self.onrelease)
        self.connect('motion_notify_event', self.onmove)

        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0
        self.rect = mp.Rectangle((0, 0), 0, 0, ec=color, fc=facecolor,
                                 alpha=alpha)
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
            print(self.message.format(xmin=xmin, xmax=xmax,
                                      ymin=ymin, ymax=ymax))

    def onmove(self, event):
        if self.pressevent is None or event.inaxes!=self.pressevent.inaxes:
            return
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.fig.canvas.draw()

