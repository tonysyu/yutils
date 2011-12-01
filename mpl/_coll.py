import numpy as np
from matplotlib import collections
from matplotlib import transforms


__all__ = ['CircleCollection']


def pts2data_transform(ax):
    """Return transform object which converts from points units to data units.

    This transform is useful for Matplotlib collections, which draw object
    sizes using points units. (There are 72 points in an inch.)
    """
    pts2pixels = 72.0 / ax.figure.dpi
    scale_x = pts2pixels * ax.bbox.width / ax.viewLim.width
    scale_y = pts2pixels * ax.bbox.height / ax.viewLim.height
    return transforms.Affine2D().scale(scale_x, scale_y)


class CircleCollection(collections.CircleCollection):
    """Circle Collection with transform that is in data coordinates.

    Unlike matplotlib's CircleCollection, the sizes passed to CircleCollection
    are radii, not areas.
    """

    def __init__(self, radii, transOffset=None, **kwargs):
        assert transOffset is None, ("transOffset is automatically set when "
                                     "the collection is added to an Axes.")
        r = np.asarray(radii)
        area = np.pi * r**2
        collections.CircleCollection.__init__(self, area, **kwargs)

    def set_axes(self, ax):
        collections.CircleCollection.set_axes(self, ax)
        # override default offset transform and use data transform
        self._offsets, self._uniform_offsets = self._uniform_offsets, None
        self._transOffset = ax.transData

    def get_transform(self):
        """Return transform scaling circle areas to data space.

        AlterThis needs to be recalculated when the axes change.
        """
        return pts2data_transform(self.axes)


class SquareCollection(collections.RegularPolyCollection):
    """Square Collection with transform that is in data coordinates.

    Unlike matplotlib's RegularPolyCollection, the sizes passed to
    SquareCollection are widths, not areas.
    """

    def __init__(self, sizes=(1.,), **kwargs):
        sizes = np.asarray(sizes)
        areas = np.pi / 2 * sizes**2
        super(SquareCollection, self).__init__(4, rotation=np.pi/4.,
                                               sizes=areas, **kwargs)

    def set_axes(self, ax):
        collections.RegularPolyCollection.set_axes(self, ax)
        # override default offset transform and use data transform
        self._offsets, self._uniform_offsets = self._uniform_offsets, None
        self._transOffset = ax.transData

    def get_transform(self):
        """Return transform scaling circle areas to data space.

        AlterThis needs to be recalculated when the axes change.
        """
        return pts2data_transform(self.axes)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    f, (ax1, ax2) = plt.subplots(ncols=2)

    c = CircleCollection([1], offsets=[(1, 1)])
    ax1.add_collection(c)
    ax1.axis([0, 2, 0, 2])
    ax1.set_aspect('equal')

    s = SquareCollection(sizes=[1], offsets=[(1, 1)])
    ax2.add_collection(s)
    ax2.axis([0, 2, 0, 2])
    ax2.set_aspect('equal')
    plt.show()

