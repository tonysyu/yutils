import warnings
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.transforms import TransformedBbox, Affine2D

__all__ = ['tight_layout', 'tight', 'tight_borders', 'tight_subplot_spacing']


PAD_INCHES = 0.04


def tight_layout(*args, **kwargs):
    warnings.warn('tight_layout function renamed tight', DeprecationWarning)
    return tight(*args, **kwargs)


def tight(pad_inches=PAD_INCHES, h_pad_inches=None, w_pad_inches=None):
    """Adjust subplot parameters to give specified padding.
    
    Parameters
    ----------
    pad_inches : float
        minimum padding between the figure edge and the edges of subplots.
    h_pad_inches, w_pad_inches : float
        minimum padding (height/width) between edges of adjacent subplots.
        Defaults to `pad_inches`.
    """
    if h_pad_inches is None:
        h_pad_inches = pad_inches
    if w_pad_inches is None:
        w_pad_inches = pad_inches

    fig = plt.gcf()
    renderer = RendererAgg(fig.get_figwidth(),
                           fig.get_figheight(),
                           fig.get_dpi())

    tight_borders(fig, renderer, pad_inches=pad_inches)
    # NOTE: border padding affects subplot spacing; tighten border first
    tight_subplot_spacing(fig, renderer, h_pad_inches, w_pad_inches)


def tight_borders(fig, renderer, pad_inches=PAD_INCHES):
    """Stretch subplot boundaries to figure edges plus padding."""
    # call draw to update the renderer and get accurate bboxes.
    fig.draw(renderer)
    bbox_original = fig.bbox_inches
    bbox_tight = fig.get_tightbbox(renderer).padded(pad_inches)
    
    # figure dimensions ordered like bbox.extents: x0, y0, x1, y1
    lengths = np.array([bbox_original.width, bbox_original.height,
                        bbox_original.width, bbox_original.height])
    whitespace = (bbox_tight.extents - bbox_original.extents) / lengths
    
    # border padding ordered like bbox.extents: x0, y0, x1, y1
    current_borders = np.array([fig.subplotpars.left, fig.subplotpars.bottom,
                                fig.subplotpars.right, fig.subplotpars.top])
    
    left, bottom, right, top = current_borders - whitespace
    fig.subplots_adjust(bottom=bottom, top=top, left=left, right=right)


def tight_subplot_spacing(fig, renderer, h_pad_inches, w_pad_inches):
    """Stretch subplots so adjacent subplots are separated by given padding."""
    # Zero hspace and wspace to make it easier to calculate the spacing.
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.draw(renderer)
    
    figbox = fig.bbox_inches
    ax_bottom, ax_top, ax_left, ax_right = _get_grid_boundaries(fig, renderer)
    nrows, ncols = ax_bottom.shape
    
    subplots_height = fig.subplotpars.top - fig.subplotpars.bottom
    if nrows > 1:
        h_overlap_inches = ax_top[1:] - ax_bottom[:-1]
        hspace_inches = h_overlap_inches.max() + h_pad_inches
        hspace_fig_frac = hspace_inches / figbox.height
        hspace = _fig_frac_to_cell_frac(hspace_fig_frac, subplots_height, nrows)
        fig.subplots_adjust(hspace=hspace)
    
    subplots_width = fig.subplotpars.right - fig.subplotpars.left
    if ncols > 1:
        w_overlap_inches = ax_right[:,:-1] - ax_left[:,1:]
        wspace_inches = w_overlap_inches.max() + w_pad_inches
        wspace_fig_frac = wspace_inches / figbox.width
        wspace = _fig_frac_to_cell_frac(wspace_fig_frac, subplots_width, ncols)
        fig.subplots_adjust(wspace=wspace)


def _get_grid_boundaries(fig, renderer):
    """Return grid boundaries for bboxes of subplots
    
    Returns
    -------
    ax_bottom, ax_top, ax_left, ax_right : array
        bbox cell-boundaries of subplot grid. If a subplot spans cells, the grid
        boundaries cutting through that subplot will be masked.
    """
    nrows, ncols, n = fig.axes[0].get_geometry()
    # Initialize boundaries as masked arrays; in the future, support subplots 
    # that span multiple rows/columns, which would have masked values for grid 
    # boundaries that cut through the subplot.
    ax_bottom, ax_top, ax_left, ax_right = [np.ma.masked_all((nrows, ncols))
                                            for n in range(4)]
    px2inches_trans = Affine2D().scale(1./fig.dpi)
    for ax in fig.axes:
        ax_bbox = ax.get_tightbbox(renderer)
        x0, y0, x1, y1 = TransformedBbox(ax_bbox, px2inches_trans).extents
        nrows, ncols, n = ax.get_geometry()
        # subplot number starts at 1, matrix index starts at 0
        i = n - 1
        ax_bottom.flat[i] = y0
        ax_top.flat[i] = y1
        ax_left.flat[i] = x0
        ax_right.flat[i] = x1
    return ax_bottom, ax_top, ax_left, ax_right


def _fig_frac_to_cell_frac(fig_frac, subplots_frac, num_cells):
    """Return fraction of cell (row/column) from a given fraction of the figure
    
    Parameters
    ----------
    fig_frac : float
        length given as a fraction of figure height or width
    subplots_frac : float
        fraction of figure (height or width) occupied by subplots
    num_cells : int
        number of rows or columns.
    """
    # This function is reverse engineered from the calculation of `sepH` and 
    # `sepW` in  `GridSpecBase.get_grid_positions`.
    return (fig_frac * num_cells) / (subplots_frac - fig_frac*(num_cells-1))


if __name__ == '__main__':
    import numpy as np
    np.random.seed(1234)
    fontsizes = [8, 16, 24, 32]
    def example_plot(ax):
        ax.plot([1, 2])
        ax.set_xlabel('x-label', fontsize=fontsizes[np.random.randint(len(fontsizes))])
        ax.set_ylabel('y-label', fontsize=fontsizes[np.random.randint(len(fontsizes))])
        ax.set_title('Title', fontsize=fontsizes[np.random.randint(len(fontsizes))])
    
    fig = plt.figure()
    example_plot(plt.subplot(1, 1, 1))
    tight()

    fig = plt.figure()
    example_plot(plt.subplot(2, 2, 1))
    example_plot(plt.subplot(2, 2, 2))
    example_plot(plt.subplot(2, 2, 3))
    example_plot(plt.subplot(2, 2, 4))
    tight()

    fig = plt.figure()
    example_plot(plt.subplot(2, 1, 1))
    example_plot(plt.subplot(2, 1, 2))
    tight()

    fig = plt.figure()
    example_plot(plt.subplot(1, 2, 1))
    example_plot(plt.subplot(1, 2, 2))
    tight()

    fig = plt.figure()
    for n in range(1, 10):
        example_plot(plt.subplot(3, 3, n))
    tight()
    plt.show()
