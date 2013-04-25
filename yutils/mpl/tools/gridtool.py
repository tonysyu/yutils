from base import BaseTool


__all__ = ['GridTool']


class GridTool(BaseTool):
    """Draw horizontal and vertical grid lines on a plot.

    Add, move, and delete horizontal and vertical lines to plot. When the tool
    is closed.

    Parameters
    ----------
    tolerance : float
        Select line if mouse click is within `tolerance` pixels.

    color : matplotlib.colors
        Line color.

    alpha : 0 < float <= 1
        Alpha value (i.e. opacity) of rectangle.

    lineweight : float
        Line weight.

    Key-bindings
    ------------
    'h' : Add horizontal line.
    'v' : Add vertical line.
    'd' : Delete *selected* line.
          (Select by clicking and holding left-mouse button).

    Attributes
    ----------
    xdata : list of floats
        x-positions of vertical lines

    ydata : list of floats
        y-positions of horizontal lines

    Notes
    -----
    This tool behaves strangely when you click the figure before it is
    completely drawn.
    """

    def __init__(self, ax, tolerance=5, color='r', alpha=0.3, lineweight=1):

        BaseTool.__init__(self, ax)

        self.tolerance = tolerance

        self.line_kwargs = dict(color=color, alpha=alpha, lw=lineweight)
        self.hlines = []
        self.vlines = []
        self.active_line = None

        self.connect('button_release_event', self.onrelease)
        self.connect('key_press_event', self.onkeypress)
        self.connect('motion_notify_event', self.onmove)
        self.connect('pick_event', self.onpick)

    @property
    def xdata(self):
        return [line.get_xdata()[0] for line in self.vlines]

    @property
    def ydata(self):
        return [line.get_ydata()[0] for line in self.hlines]

    def onpick(self, event):
        self.active_line = event.artist

    def onrelease(self, event):
        if event.button == 1:
            self.active_line = None

    def onmove(self, event):
        ignore = (self.active_line is None or
                  not event.inaxes or
                  event.button != 1)
        if ignore:
            return

        if self.active_line in self.hlines:
            self.active_line.set_ydata([event.ydata, event.ydata])
        elif self.active_line in self.vlines:
            self.active_line.set_xdata([event.xdata, event.xdata])
        else:
            raise Exception("Unregistered artist selected")

        self.canvas.draw()

    def onkeypress(self, event):
        if not event.inaxes:
            return

        if event.key=='v':
            self.add_vline(event.xdata)
        elif event.key=='h':
            self.add_hline(event.ydata)
        elif event.key=='d':
            if self.active_line is None:
                return
            raise NotImplementedError

        self.canvas.draw()

    def add_hline(self, y):
        line = self.ax.axhline(y, **self.line_kwargs)
        line.set_picker(self.tolerance)
        self.hlines.append(line)

    def add_vline(self, x):
        line = self.ax.axvline(x, **self.line_kwargs)
        line.set_picker(self.tolerance)
        self.vlines.append(line)

