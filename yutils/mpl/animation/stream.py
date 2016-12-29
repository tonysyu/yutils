import numpy as np
import matplotlib.pyplot as plt

from matplotlib import mlab
from scipy import integrate
from scipy import interpolate

try:
    from mpltools.animation import Animation
except ImportError as error:
    msg = "`yutils.mpl.animation.stream` not available: `mpltools` not found"
    raise ImportError(msg)

try:
    import sympy
    from sympy.abc import x, y
    sympy_available = True
except ImportError:
    sympy_available = False


plt.rc('contour', negative_linestyle='solid')


def velocity_functions(method='field', *args, **kwargs):
    """Return velocity functions from velocity fields or potential function.

    Parameters
    ----------
    method : {'field', 'potential', 'stream'}
        Input method for generating velocity functions. The positional and
        keyword arguments will depend the method.

        - 'field'
          2D field of coordinates and velocities: X, Y, U, V.
          See `velocity_functions_from_field` for details::

            u, v = velocity_functions(X, Y, U, V, method='field')

        - 'potential'
          SymPy equation specifying potential function for flow.
          See `velocity_functions_from_potential` for details::

            u, v = velocity_functions(phi, method='potential')

        - 'stream'
          SymPy equation specifying stream function for flow.
          See `velocity_functions_from_potential` for details::

            u, v = velocity_functions(psi, method='stream')

    """
    if method == 'field':
        velocity_functions_from_field(*args, **kwargs)
    elif method in ('potential', 'stream'):
        velocity_functions_from_potential(*args, **kwargs)
    else:
        raise ValueError("Unrecognized method: %s" % method)


def velocity_functions_from_potential(potential, method='potential'):

    if not sympy_available:
        msg = "SymPy required to calculate velocity funcs from %s function"
        raise ImportError(msg % method)

    if method == 'potential':
        phi = potential
        u = sympy.lambdify((x, y), phi.diff(x), 'numpy')
        v = sympy.lambdify((x, y), phi.diff(y), 'numpy')
    elif method == 'stream':
        psi = potential
        u = sympy.lambdify((x, y), psi.diff(y), 'numpy')
        v = sympy.lambdify((x, y), -psi.diff(x), 'numpy')
    return u, v


def velocity_functions_from_field(X, Y, U, V, interp='cubic'):
    u = interpolate.interp2d(X, Y, U, kind=interp)
    v = interpolate.interp2d(X, Y, V, kind=interp)
    return u, v


def displacements_from_velocity_funcs(u_func, v_func, method='euler'):

    def velocity(xy, t=0):
        u = u_func(xy[0], xy[1])
        v = v_func(xy[0], xy[1])
        # Note the return value must be a list (not a tuple) for scipy's
        # integrate function to work properly
        return [u, v]

    def euler_forward(f, pts, dt):
        vel = np.asarray([f(p) for p in pts])
        return pts + vel * dt

    def rk4(f, pts, dt):
        new_pts = [mlab.rk4(f, p, [0, dt])[-1] for p in pts]
        return new_pts

    def vode(f, pts, dt):
        new_pts = [integrate.odeint(f, p, [0, dt])[-1] for p in pts]
        return new_pts

    integrator = dict(euler=euler_forward, rk4=rk4, scipy=vode)[method]

    def displace(pts, dt=1):
        pts = np.asarray(pts)
        return integrator(velocity, pts, dt)

    return displace


def plot_points(pts, *args, **kwargs):
    ax = kwargs.pop('ax', plt.gca())
    x, y = np.asarray(pts).transpose()
    return ax.plot(x, y, *args, **kwargs)


def remove_points(pts, xlim, ylim):
    if len(pts) == 0:
        return []
    pts = np.asarray(pts)
    outside_xlim = (pts[:, 0] < xlim[0]) | (pts[:, 0] > xlim[1])
    outside_ylim = (pts[:, 1] < ylim[0]) | (pts[:, 1] > ylim[1])
    keep = ~(outside_xlim|outside_ylim)
    return pts[keep]

def random_y(ylim):
    yrange = np.diff(ylim)
    return yrange * np.random.rand(1)[0] + ylim[0]

def get_psi_limits(Z):
    """Return (min, max) psi values based on edges of domain

    Use values on edge to avoid singularities in the domain (assuming the
    singularities aren't close to the edge.
    """
    psi = [Z[0,:], Z[-1, :], Z[:, 0], Z[:, -1]]
    return np.min(psi), np.max(psi)


def plot_streamlines(u, v, xlim, ylim, ax):
    x0, x1 = xlim
    y0, y1 = ylim
    Y, X =  np.mgrid[x0:x1:100j, y0:y1:100j]
    U = u(X, Y)
    V = v(X, Y)
    ax.streamplot(X, Y, U, V, color='0.7')


def flow_profile(velocity, span=1., num_vecs=10, orientation='horizontal'):
    """Plot velocity profile from a velocity function.

    Parameters
    ----------
    velocity : function
        Velocity function taking a single coordinate from 0 to `span`.
    span : float
        Span of velocity profile.
    num_vecs : int
        Number of velocity vectors plotted.

    Returns
    -------
    vecs : :class:`~class.matplotlib.collections.LineCollection`
        Collection of velocity vectors.
    profile : :class:`~matplotlib.lines.Line2D`
        Curve representing velocity profile.
    zero_line : :class:`~matplotlib.lines.Line2D`
        Line marking zero velocity.
    """
    z_vecs = span * np.linspace(0, 1, num_vecs)
    u_vecs = velocity(z_vecs)
    if orientation == 'horizontal':
        vecs = plt.hlines(z_vecs, xmin=0, xmax=u_vecs, linewidth=1, color='k')
    elif orientation == 'vertical':
        vecs = plt.vlines(z_vecs, ymin=0, ymax=u_vecs, linewidth=1, color='k')
    else:
        raise ValueError("Unknown orientation: %s" % orientation)

    z = span * np.linspace(0, 1.)
    U = velocity(z)

    x_zero = [0, 0]
    y_zero = [min(z), max(z)]
    x_prof = U
    y_prof = z
    if orientation == 'vertical':
        x_zero, y_zero = y_zero, x_zero
        x_prof, y_prof = y_prof, x_prof

    zero_line = plt.plot(x_zero, y_zero, 'k--', linewidth=1)
    profile = plt.plot(x_prof, y_prof)

    return vecs, profile, zero_line


class StreamFuncAnim(Animation):

    def __init__(self, stream_function, xlim=(-1, 1), ylim=None):
        self.stream_function = stream_function
        self.xlim = xlim
        self.ylim = ylim if ylim is not None else xlim
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')

    def init_background(self):
        self.u, self.v = velocity_functions_from_potential(stream_function,
                                                           method='stream')
        plot_streamlines(self.u, self.v, self.xlim, self.ylim, self.ax)
        self.displace = displacements_from_velocity_funcs(self.u, self.v,
                                                          method='scipy')

    def update(self):
        pts = []
        while True:
            pts = list(pts)
            pts.append((self.xlim[0], random_y(self.ylim)))
            pts = self.displace(pts, 0.05)
            pts = remove_points(pts, self.xlim, self.ylim)
            self.ax.lines = []
            lines, = plot_points(pts, 'ro', ax=self.ax)
            yield lines, # return line so that blit works properly


if __name__ == '__main__':
    u = lambda x: 3/4. * x**2 - 0.5 * x
    flow_profile(u)

    def cylinder_stream_function(U=1, R=1):
        r = sympy.sqrt(x**2 + y**2)
        psi = U * (r - R**2 / r) * sympy.sin(sympy.atan2(y, x))
        return psi

    class CylinderFlow(StreamFuncAnim):
        def init_background(self):
            StreamFuncAnim.init_background(self)
            c = plt.Circle((0, 0), radius=1, facecolor='none')
            self.ax.add_patch(c)

    stream_function = cylinder_stream_function()
    cylinder_flow = CylinderFlow(stream_function, xlim=(-3, 3))
    cylinder_flow.animate(blit=True)
    plt.show()

