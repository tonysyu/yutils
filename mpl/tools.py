import numpy as np
import matplotlib.pyplot as plt


__all__ = ['MeasureLengthTool', 'RectangularSelection']


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


class RectangularSelection(object):
    def __init__(self, ax, color='k'):
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
        msg = 'Selected extents: xmin=%g, xmax=%g, ymin=%g, ymax=%g'
        print msg % self.extents

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

