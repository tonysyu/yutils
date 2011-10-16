import numpy as np


__all__ = ['MeasureLengthTool']


class MeasureLengthTool(object):
    def __init__(self, ax, color='k'):
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
        print 'Selected length: %s pixels' % self.length

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

