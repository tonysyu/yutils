import numpy as np
import matplotlib.pyplot as plt

from yutils.scaling import loglog


def slope_marker(origin, slope, size_frac=0.1, pad_frac=0.1, ax=None,
                 orientation='normal'):
    """Plot triangular slope marker labeled with slope.
    
    Parameters
    ----------
    origin : (x, y)
        tuple of x, y coordinates for the slope
    slope : float or (rise, run)
        the length of the slope triangle
    size_frac : float
        the fraction of the xaxis length used to determine the size of the slope
        marker. Should be less than 1.
    pad_frac : float
        the fraction of the slope marker used to pad text labels. Should be less 
        than 1.
    orientation : {normal|inverted}
        Normally, the slope marker is below a line for positive slopes and above
        a line for negative slopes; `orientation='inverted'` flips the marker.
    """
    if ax is None:
        ax = plt.gca()
        
    if np.iterable(slope):
        rise, run = slope
        slope = float(rise) / run
    else:
        rise = run = None
    
    xlim = ax.get_xlim()
    dx_linear = size_frac * (xlim[1] - xlim[0])
    dx_decades = size_frac * (np.log10(xlim[1]) - np.log10(xlim[0]))
    
    
    if orientation == 'inverted':
        dx_linear = -dx_linear
        dx_decades = -dx_decades

    if ax.get_xscale() == 'log':
        log_size = dx_decades
        dx = loglog.displace(origin[0], log_size) - origin[0]
        x_text = loglog.displace(origin[0], log_size/2.)
    else:
        dx = dx_linear
        x_text = origin[0] + dx/2.
    
    if ax.get_yscale() == 'log':
        log_size = dx_decades * slope
        dy = loglog.displace(origin[1], log_size) - origin[1]
        y_text = loglog.displace(origin[1], log_size/2.)
    else:
        dy = dx_linear * slope
        y_text = origin[1] + dy/2.
        
    x_pad = pad_frac * dx
    y_pad = pad_frac * dy
    va = 'top' if y_pad > 0 else 'bottom'
    ax.text(x_text, origin[1]-y_pad, str(run), va=va, ha='center')
    ha = 'left' if x_pad > 0 else 'right'
    ax.text(origin[0]+dx+x_pad, y_text, str(rise), ha=ha, va='center')
    
    ax.add_patch(_slope_triangle(origin, dx, dy))


def _slope_triangle(origin, dx, dy, ec='none', fc='0.8', **poly_kwargs):
    """Return Polygon representing slope.
          /|
         / | dy
        /__|
         dx
    """
    verts = [np.asarray(origin)]
    verts.append(verts[0] + (dx, 0))
    verts.append(verts[0] + (dx, dy))
    return plt.Polygon(verts, ec=ec, fc=fc, **poly_kwargs)


if __name__ == '__main__':
    plt.plot([0, 2], [0, 1])
    slope_marker((1, 0.4), (1, 2))
    
    x = np.logspace(0, 2)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    
    ax1.loglog(x, x**0.5)
    slope_marker((10, 2), (1, 2), ax=ax1)
    
    ax2.loglog(x, x**-0.5)
    slope_marker((10, .4), (-1, 2), ax=ax2)
    
    ax3.loglog(x, x**0.5)
    slope_marker((10, 4), (1, 2), orientation='inverted', ax=ax3)
    
    plt.show()
