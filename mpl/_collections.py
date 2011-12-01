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
    """Circle Collection with transform that is in data coordinates"""

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    f, ax = plt.subplots()

    c = CircleCollection([1], offsets=[(1, 1)])
    ax.add_collection(c)
    ax.axis([0, 2, 0, 2])
    ax.set_aspect('equal')

    plt.show()

