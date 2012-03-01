
__all__ = ['BaseTool']


class BaseTool(object):
    """Base class for tools.

    This class simplifies the clean up of event callbacks by defining a
    `disconnect` method to remove callbacks.

    Note that callbacks should be initialized with the `connect` method instead
    of `figure.canvas.mpl_connect`.

    Attributes
    ----------
    ax : matplotlib.axes.Axes
    fig : matplotlib.figure.Figure
    canvas : matplotlib figure canvas
    cids : list of callbacks
    """

    def __init__(self, ax, *args, **kwargs):
        self.ax = ax
        self.fig = ax.figure
        self.canvas = ax.figure.canvas
        self.cids = []

    def connect(self, event, callback):
        cid = self.canvas.mpl_connect(event, callback)
        self.cids.append(cid)

    def disconnect(self):
        for c in self.cids:
            self.canvas.mpl_disconnect(c)

